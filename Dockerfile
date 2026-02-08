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
    mkdir -p /content/MuseTalk/models/musetalkV15 && \
    huggingface-cli download TMElyralab/MuseTalk musetalkV15/musetalk.json \
      --local-dir /content/MuseTalk/models --local-dir-use-symlinks False && \
    huggingface-cli download TMElyralab/MuseTalk musetalkV15/unet.pth \
      --local-dir /content/MuseTalk/models --local-dir-use-symlinks False

# GFPGAN for optional face enhancement
RUN pip install --no-cache-dir gfpgan && \
    mkdir -p /content/models/gfpgan && \
    wget -q -O /content/models/gfpgan/GFPGANv1.4.pth \
      https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

# Copy our handler
COPY handler.py /content/handler.py

CMD ["python", "-u", "/content/handler.py"]
