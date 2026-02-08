"""
RunPod Serverless Handler for MuseTalk 1.5
Video + audio → lip-synced video uploaded to R2 (or base64 fallback)
Optional GFPGAN face enhancement via `enhance: true`

Uses MuseTalk's inference script as subprocess (robust against API changes).
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
import json
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("musetalk15")

# ── Paths inside camenduru image ──────────────────────────────────
MUSETALK_DIR = "/content/MuseTalk"
MODELS_DIR = "/content/MuseTalk/models"
GFPGAN_MODEL = "/content/models/gfpgan/GFPGANv1.4.pth"

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


# ── Helpers ───────────────────────────────────────────────────────

def download_file(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    return os.path.getsize(dest_path)


def get_audio_duration(audio_path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


def run_musetalk(video_path, audio_path, result_dir, bbox_shift=0):
    """Run MuseTalk 1.5 via subprocess (uses official inference script)."""
    os.makedirs(result_dir, exist_ok=True)

    # Write inference config YAML (must be dict keyed by task name, not list)
    config = {
        "task_0": {
            "video_path": video_path,
            "audio_path": audio_path,
            "bbox_shift": bbox_shift,
        }
    }
    config_path = os.path.join(result_dir, "inference_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    output_name = "output.mp4"

    cmd = [
        "python", "-m", "scripts.inference",
        "--inference_config", config_path,
        "--result_dir", result_dir,
        "--unet_model_path", os.path.join(MODELS_DIR, "musetalkV15", "unet.pth"),
        "--unet_config", os.path.join(MODELS_DIR, "musetalkV15", "musetalk.json"),
        "--version", "v15",
        "--output_vid_name", output_name,
        "--batch_size", "8",
        "--use_float16",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=MUSETALK_DIR, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"MuseTalk inference failed:\nstderr: {proc.stderr[-2000:]}\nstdout: {proc.stdout[-2000:]}")

    # Find output video
    output_path = os.path.join(result_dir, output_name)
    if not os.path.exists(output_path):
        # Search for any mp4 in result_dir
        for root, dirs, files in os.walk(result_dir):
            for f in files:
                if f.endswith('.mp4'):
                    output_path = os.path.join(root, f)
                    break

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"No output video found in {result_dir}\nstdout: {proc.stdout[-2000:]}")

    duration = get_audio_duration(audio_path)
    return output_path, duration


def _enhance_video(input_path, output_path):
    """Post-process video with GFPGAN face restoration."""
    import cv2
    import torch
    import sys
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
    from gfpgan import GFPGANer

    restorer = GFPGANer(
        model_path=GFPGAN_MODEL,
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        device='cuda',
    )

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
        _, _, enhanced = restorer.enhance(
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
    del restorer
    torch.cuda.empty_cache()
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

        output_path, duration = run_musetalk(
            video_path, audio_path, result_dir, bbox_shift
        )
        inf_time = time.time() - t0

        # Optional GFPGAN face enhancement
        enh_time = 0
        if enhance and os.path.exists(GFPGAN_MODEL):
            logger.info(f"[{job_id}] Running GFPGAN enhancement...")
            enhanced_path = str(workdir / "enhanced.mp4")
            _, enh_time = _enhance_video(output_path, enhanced_path)
            os.remove(output_path)
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
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ── Start ─────────────────────────────────────────────────────────
logger.info("MuseTalk 1.5 RunPod handler ready!")
runpod.serverless.start({"handler": handler})
