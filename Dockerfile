###############################################################################
# Make-It-Animatable  •  Inference-only GPU image for Google Cloud Run
#
#   • CUDA 12.4 runtime (works on NVIDIA L4 / A100)
#   • Python 3.11, FastAPI, Uvicorn, Google-Cloud-Storage
#   • Blender 4.2 headless (for vis_blender)
#   • Downloads pretrained weights `output/best/new/*`
#   • Final image ≈ 4 GB  (single-stage for clarity; multi-stage possible)
###############################################################################

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ───── Basic env  ────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PORT=8080

# ───── System deps  ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-venv \
        git git-lfs wget ca-certificates \
        xvfb libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ───── Workdir & repo  ──────────────────────────────────────────────────────
WORKDIR /app
COPY . /app                         # assumes Docker build context is repo root

# ───── Python deps  ─────────────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install -r requirements-demo.txt \
                fastapi uvicorn[standard] \
                google-cloud-storage "pydantic>=2"

# ───── Blender 4.2 headless  ────────────────────────────────────────────────
# keep it slim: no GUI libs, just off-screen rendering
RUN wget -qO blender.tar.xz https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz && \
    tar -xf blender.tar.xz && mv blender-* /opt/blender && \
    rm blender.tar.xz
ENV PATH="/opt/blender:$PATH"

# ───── Pull pretrained weights (LFS)  ───────────────────────────────────────
RUN git lfs install && \
    git -C /app lfs pull -I output/best/new

# ───── Default runtime env  ─────────────────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OUTPUT_BUCKET=mia-results        # set or override at deploy time

# ───── Entrypoint  ──────────────────────────────────────────────────────────
# one worker ⇒ one job at a time; Cloud Run can autoscale to more GPU pods
CMD ["uvicorn", "serve_mia:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
