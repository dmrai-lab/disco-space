# The DiSCo Space image: python 3.11, the pinned stack (requirements.txt), the app on port 7860.
# Built by Hugging Face on every push to the Space repo (no GPU at build time: JAX compiles at first use, into
# JAX_COMPILATION_CACHE_DIR, which lives on the Space's persistent storage when one is mounted at /data).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_COMPILATION_CACHE_DIR=/data/jax-cache \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.5 \
    HF_HOME=/data/hf \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# the CUDA wheels' libraries, or the plugin falls back to the CPU silently
RUN python - <<'EOF'
import glob, os, site
libs = sorted(set(os.path.dirname(p) for sp in site.getsitepackages() for p in glob.glob(os.path.join(sp, "nvidia", "*", "lib"))))
open("/etc/ld.so.conf.d/nvidia-wheels.conf", "w").write("\n".join(libs) + "\n")
EOF
RUN ldconfig

COPY --chown=user space /app/space
COPY --chown=user data /app/data
COPY --chown=user entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && mkdir -p /data && chown user /data

USER user
EXPOSE 7860
CMD ["/app/entrypoint.sh"]
