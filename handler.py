"""
RunPod Serverless Handler for MuseTalk 1.5
Video + audio → lip-synced video uploaded to R2 (or base64 fallback)
Optional GFPGAN face enhancement via `enhance: true`
"""

import runpod
import os
import sys
import time
import subprocess
import base64
import shutil
import uuid
import gc
import urllib.request
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("musetalk15")

# ── Paths inside camenduru image ──────────────────────────────────
MUSETALK_DIR = "/content/MuseTalk"
MODELS_DIR = "/content/MuseTalk/models"

sys.path.insert(0, MUSETALK_DIR)
os.chdir(MUSETALK_DIR)

# ── R2 / S3 Config ───────────────────────────────────────────────
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "tray")

s3_client = None
if R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    import boto3
    from botocore.config import Config
    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    logger.info(f"R2 client configured: bucket={R2_BUCKET_NAME}")
else:
    logger.warning("R2 not configured — will return base64 output")

# ── Preload MuseTalk 1.5 models ──────────────────────────────────
logger.info("Loading MuseTalk 1.5 models...")
t_load = time.time()

import torch
import numpy as np
import cv2
import copy

from musetalk.utils.utils import load_all_model

# v1.5 paths (load_all_model defaults to v1.5 in latest code)
V15_UNET = os.path.join(MODELS_DIR, "musetalkV15", "unet.pth")
V15_CONFIG = os.path.join(MODELS_DIR, "musetalkV15", "musetalk.json")

audio_processor, vae, unet, pe = load_all_model(
    unet_model_path=V15_UNET,
    unet_config=V15_CONFIG,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import datagen

logger.info(f"MuseTalk 1.5 models loaded in {time.time() - t_load:.1f}s")

# ── Preload GFPGAN (optional face enhancement) ───────────────────
GFPGAN_MODEL = "/content/models/gfpgan/GFPGANv1.4.pth"
gfpgan_restorer = None
if os.path.exists(GFPGAN_MODEL):
    from gfpgan import GFPGANer
    gfpgan_restorer = GFPGANer(
        model_path=GFPGAN_MODEL,
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        device=device,
    )
    logger.info("GFPGAN loaded for face enhancement")


# ── Helpers ───────────────────────────────────────────────────────

def download_file(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    return os.path.getsize(dest_path)


def run_musetalk(video_path, audio_path, result_dir, bbox_shift=0):
    """Run MuseTalk 1.5 lip-sync on video + audio."""
    os.makedirs(result_dir, exist_ok=True)

    # Resample video to 25fps
    resampled = os.path.join(result_dir, "input_25fps.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-r", "25", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", resampled],
        check=True, capture_output=True,
    )

    # Normalize audio to 16kHz mono WAV
    audio_norm = os.path.join(result_dir, "audio_16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", audio_norm],
        check=True, capture_output=True,
    )

    # Get audio duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_norm],
        capture_output=True, text=True, check=True,
    )
    audio_duration = float(probe.stdout.strip())
    logger.info(f"Audio duration: {audio_duration:.2f}s")

    # Extract frames
    frames_dir = os.path.join(result_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", resampled, "-q:v", "2", f"{frames_dir}/%06d.png"],
        check=True, capture_output=True,
    )

    input_img_list = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.endswith('.png')
    ])
    logger.info(f"Extracted {len(input_img_list)} frames")

    if not input_img_list:
        raise ValueError("No frames extracted from video")

    # Loop frames to match audio duration (ping-pong)
    fps = 25
    needed_frames = int(audio_duration * fps) + 1
    if len(input_img_list) < needed_frames:
        extended = []
        forward = True
        idx = 0
        while len(extended) < needed_frames:
            extended.append(input_img_list[idx])
            if forward:
                idx += 1
                if idx >= len(input_img_list):
                    forward = False
                    idx = len(input_img_list) - 2
            else:
                idx -= 1
                if idx < 0:
                    forward = True
                    idx = 1
        input_img_list = extended
        logger.info(f"Looped to {len(input_img_list)} frames")

    # Read frames and detect faces
    frame_list_cycle = read_imgs(input_img_list[:needed_frames])
    coord_list, frame_list = get_landmark_and_bbox(input_img_list[:needed_frames], bbox_shift)

    for i, c in enumerate(coord_list):
        if c is None:
            coord_list[i] = coord_placeholder

    # Process audio features
    logger.info("Processing audio features...")
    whisper_feature = audio_processor.audio2feat(audio_norm)
    whisper_chunks = audio_processor.feature2chunks(
        feature_array=whisper_feature, fps=fps
    )
    logger.info(f"Audio chunks: {len(whisper_chunks)}")

    # Run inference
    logger.info("Running MuseTalk 1.5 inference...")
    t_inf = time.time()
    gen = datagen(whisper_chunks, frame_list_cycle, batch_size=8, delay_frame=0)

    res_frame_list = []
    idx = 0
    for i, (whisper_batch, frame_batch) in enumerate(gen):
        tensor_list = [
            torch.Tensor(np.array(f).astype(np.float32)).to(device)
            for f in frame_batch
        ]
        with torch.no_grad():
            frames_tensor = torch.stack(tensor_list, dim=0).permute(0, 3, 1, 2) / 255.0
            latents = vae.encode(frames_tensor).latent_dist.sample()
            latents = latents * 0.18215

            audio_tensor = torch.Tensor(whisper_batch).to(device)
            audio_feature_batch = pe(audio_tensor)

            pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature_batch).sample

            recon = vae.decode(pred_latents / 0.18215).sample
            recon = (recon.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

        for j, res_frame in enumerate(recon):
            bbox = coord_list[idx % len(coord_list)]
            ori_frame = copy.deepcopy(frame_list_cycle[idx % len(frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            res_face = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            combine_frame = get_image(ori_frame, res_face, bbox)
            res_frame_list.append(combine_frame)
            idx += 1

        if (i + 1) % 20 == 0:
            logger.info(f"  Batch {i+1}, frames: {idx}/{needed_frames}")

    inf_time = time.time() - t_inf
    logger.info(f"Inference: {len(res_frame_list)} frames in {inf_time:.1f}s ({len(res_frame_list)/inf_time:.1f} fps)")

    # Write output video
    output_video = os.path.join(result_dir, "output_no_audio.mp4")
    output_final = os.path.join(result_dir, "output.mp4")

    import imageio
    writer = imageio.get_writer(output_video, fps=fps, codec='libx264', pixelformat='yuv420p')
    for frame in res_frame_list:
        writer.append_data(frame)
    writer.close()

    # Mux audio
    subprocess.run(
        ["ffmpeg", "-y", "-i", output_video, "-i", audio_norm,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-shortest", output_final],
        check=True, capture_output=True,
    )

    return output_final, audio_duration, inf_time


def enhance_video(input_path, output_path):
    """Post-process video with GFPGAN face enhancement."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video = output_path + '.tmp.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

    count = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, _, enhanced = gfpgan_restorer.enhance(
            frame, has_aligned=False, only_center_face=True, paste_back=True
        )
        writer.write(enhanced)
        count += 1
        if count % 100 == 0:
            logger.info(f"  Enhanced {count} frames")

    cap.release()
    writer.release()
    enh_time = time.time() - t0

    # Re-encode with h264 + mux audio from original
    subprocess.run([
        'ffmpeg', '-y',
        '-i', temp_video, '-i', input_path,
        '-map', '0:v', '-map', '1:a?',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-c:a', 'copy', '-pix_fmt', 'yuv420p',
        output_path
    ], capture_output=True, check=True)

    os.remove(temp_video)
    logger.info(f"Enhanced {count} frames in {enh_time:.1f}s ({count/enh_time:.1f} fps)")
    return count, enh_time


# ── RunPod handler ────────────────────────────────────────────────

def handler(event):
    workdir = None
    try:
        inp = event.get("input", {})
        video_url = inp.get("video_url")
        audio_url = inp.get("audio_url")
        bbox_shift = inp.get("bbox_shift", 0)
        return_base64 = inp.get("return_base64", False)
        enhance = inp.get("enhance", False)

        job_id = str(uuid.uuid4())[:8]
        workdir = Path(tempfile.mkdtemp(prefix=f"mt15_{job_id}_"))

        # Download inputs
        video_path = str(workdir / "input.mp4")
        audio_path = str(workdir / "input_audio.wav")

        if video_url:
            logger.info(f"[{job_id}] Downloading video...")
            vsize = download_file(video_url, video_path)
            logger.info(f"[{job_id}] Video: {vsize/1024:.1f} KB")
        else:
            return {"error": "Must provide video_url"}

        if audio_url:
            logger.info(f"[{job_id}] Downloading audio...")
            asize = download_file(audio_url, audio_path)
            logger.info(f"[{job_id}] Audio: {asize/1024:.1f} KB")
        else:
            return {"error": "Must provide audio_url"}

        # Run MuseTalk 1.5
        result_dir = str(workdir / "results")
        logger.info(f"[{job_id}] Running MuseTalk 1.5 (enhance={enhance})...")
        t0 = time.time()

        output_path, duration, inf_time = run_musetalk(
            video_path, audio_path, result_dir, bbox_shift
        )

        # Optional GFPGAN face enhancement
        enh_time = 0
        if enhance and gfpgan_restorer is not None:
            logger.info(f"[{job_id}] Running GFPGAN enhancement...")
            enhanced_path = str(workdir / "enhanced.mp4")
            _, enh_time = enhance_video(output_path, enhanced_path)
            output_path = enhanced_path

        total_time = time.time() - t0
        output_size = os.path.getsize(output_path)
        logger.info(f"[{job_id}] Done in {total_time:.1f}s (inference: {inf_time:.1f}s, enhance: {enh_time:.1f}s, output: {output_size/1024:.1f} KB)")

        # Return result
        result = {
            "job_id": job_id,
            "duration_seconds": duration,
            "inference_time": round(inf_time, 1),
            "enhance_time": round(enh_time, 1),
            "enhanced": enhance,
            "processing_time_seconds": round(total_time, 1),
            "output_size_kb": round(output_size / 1024, 1),
        }

        if s3_client and not return_base64:
            s3_key = f"musetalk/{job_id}.mp4"
            s3_client.upload_file(
                output_path, R2_BUCKET_NAME, s3_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": s3_key},
                ExpiresIn=86400,
            )
            result["video_url"] = presigned_url
            logger.info(f"[{job_id}] Uploaded to R2: {s3_key}")
        else:
            with open(output_path, "rb") as f:
                result["video_base64"] = base64.b64encode(f.read()).decode()
            logger.info(f"[{job_id}] Returning base64")

        return result

    except Exception as e:
        logger.exception("Handler failed")
        return {"error": str(e)}

    finally:
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Start ─────────────────────────────────────────────────────────
logger.info("MuseTalk 1.5 RunPod worker ready!")
runpod.serverless.start({"handler": handler})
