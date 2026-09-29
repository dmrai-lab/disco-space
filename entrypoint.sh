#!/bin/bash
# The Space's entry: caches on the persistent volume when /data is writable, else on the ephemeral disk.
set -e
if ! ( mkdir -p /data/jax-cache /data/hf 2>/dev/null && touch /data/.w 2>/dev/null ); then
  export JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache HF_HOME=/tmp/hf
  mkdir -p /tmp/jax-cache /tmp/hf
fi
python -c "import jax; print('devices', jax.devices(), flush=True)"
exec python -m space.app
