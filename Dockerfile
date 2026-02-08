# MuseTalk 1.5 on RunPod Serverless
# Base: camenduru image (has v1.0 weights + all deps: torch, mmcv, etc.)
# We add: v1.5 weights + updated MuseTalk code + R2 upload support
FROM camenduru/style-tts-muse-talk:latest

# Install runpod SDK + deps
RUN pip install --no-cache-dir runpod boto3 pyyaml

# Update MuseTalk to latest (has v1.5 support in load_all_model)
RUN cd /content/MuseTalk && git pull origin main

# Download MuseTalk v1.5 weights from HuggingFace
RUN pip install -q huggingface-hub && \
    huggingface-cli download TMElyralab/MuseTalk musetalkV15/musetalk.json \
      --local-dir /content/MuseTalk/models --local-dir-use-symlinks False && \
    huggingface-cli download TMElyralab/MuseTalk musetalkV15/unet.pth \
      --local-dir /content/MuseTalk/models --local-dir-use-symlinks False

# Download SD-VAE (stabilityai/sd-vae-ft-mse) into models/sd-vae/
RUN mkdir -p /content/MuseTalk/models/sd-vae && \
    huggingface-cli download stabilityai/sd-vae-ft-mse config.json \
      --local-dir /content/MuseTalk/models/sd-vae --local-dir-use-symlinks False && \
    huggingface-cli download stabilityai/sd-vae-ft-mse diffusion_pytorch_model.bin \
      --local-dir /content/MuseTalk/models/sd-vae --local-dir-use-symlinks False

# Download Whisper-tiny for audio processing (required by v1.5)
RUN mkdir -p /content/MuseTalk/models/whisper && \
    huggingface-cli download openai/whisper-tiny config.json \
      --local-dir /content/MuseTalk/models/whisper --local-dir-use-symlinks False && \
    huggingface-cli download openai/whisper-tiny pytorch_model.bin \
      --local-dir /content/MuseTalk/models/whisper --local-dir-use-symlinks False && \
    huggingface-cli download openai/whisper-tiny preprocessor_config.json \
      --local-dir /content/MuseTalk/models/whisper --local-dir-use-symlinks False

# GFPGAN for optional face enhancement
RUN pip install --no-cache-dir gfpgan && \
    mkdir -p /content/models/gfpgan && \
    wget -q -O /content/models/gfpgan/GFPGANv1.4.pth \
      https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

# Fix basicsr/torchvision compatibility (functional_tensor removed in newer torchvision)
RUN FPATH=$(python -c "import torchvision, os; print(os.path.join(os.path.dirname(torchvision.__file__), 'transforms', 'functional_tensor.py'))") && \
    rm -f "$FPATH" && \
    echo "from torchvision.transforms.functional import *" > "$FPATH"

# Copy our handler
COPY handler.py /content/handler.py

CMD ["python", "-u", "/content/handler.py"]
