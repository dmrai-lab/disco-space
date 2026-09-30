#!/bin/bash
# The Space's entry: the caches named by the image's environment (JAX_COMPILATION_CACHE_DIR, HF_HOME under /data, the
# container's own disk) are created, then the page serves; it downloads the layout from the Hub while it comes up.
set -e
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$HF_HOME"
exec python -m space.app
