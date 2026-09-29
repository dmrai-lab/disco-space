"""The acceptance gate as a Hugging Face Job: a GPU container fetches the Space repository's files at one revision
(what is deployed, tests included), installs the pinned requirements, runs ``tests/test_acceptance.py`` and prints
``gate.json``; this script starts the job, waits, prints its log and exits with the job's outcome. The Space is
private, so the container gets the caller's Hub token as the job secret ``HF_TOKEN``.

    python tools/gate_job.py [--space rfick/disco] [--revision <space commit>] [--flavor l4x1] [--timeout 45m]
"""
import argparse
import os
import sys
import time

SCRIPT = r'''
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > /dev/null && apt-get install -y -qq git > /dev/null
pip install -q "huggingface_hub>=1.0"
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(os.environ["SPACE"], repo_type="space", revision=os.environ.get("REVISION") or None, local_dir="/w")
print("files at", p)
EOF
cd /w
pip install -q pytest -r requirements.txt
python - <<'EOF'
import glob, os, site
libs = sorted(set(os.path.dirname(p) for sp in site.getsitepackages() for p in glob.glob(os.path.join(sp, "nvidia", "*", "lib"))))
open("/etc/ld.so.conf.d/nvidia-wheels.conf", "w").write("\n".join(libs) + "\n")
EOF
ldconfig
python -c "import jax; print('devices', jax.devices())"
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true pytest tests/test_acceptance.py -s -q -p no:cacheprovider
echo "=== gate.json"; cat gate.json; echo; echo "=== GATE PASSED"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=os.environ.get("DISCO_SPACE", "rfick/disco"))
    ap.add_argument("--revision", default=None)
    ap.add_argument("--flavor", default="l4x1")
    ap.add_argument("--timeout", default="45m")
    ap.add_argument("--image", default="python:3.11-slim")
    a = ap.parse_args()
    from huggingface_hub import HfApi, get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)
    rev = a.revision or api.repo_info(a.space, repo_type="space").sha
    job = api.run_job(image=a.image, command=["bash", "-c", SCRIPT], env={"SPACE": a.space, "REVISION": rev},
                      secrets={"HF_TOKEN": token}, flavor=a.flavor, timeout=a.timeout, name=f"disco-space-gate-{rev[:8]}")
    print(f"job {job.id} on {a.flavor}: {a.space}@{rev}", flush=True)
    t0 = time.time()
    done = api.wait_for_job(job.id, poll_interval=15)
    stage = done.status.stage
    print(f"job {job.id} {stage} after {time.time() - t0:.0f} s", flush=True)
    log = "\n".join(api.fetch_job_logs(job_id=job.id))
    print(log[-20000:])
    ok = stage == "COMPLETED" and "=== GATE PASSED" in log
    print("GATE", "PASSED" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
