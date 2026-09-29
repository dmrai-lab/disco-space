"""The acceptance gate as a Hugging Face Job: a GPU container clones this repository at one commit, installs its
pinned requirements, runs ``tests/test_acceptance.py`` and prints ``gate.json``; this script starts the job, waits,
prints its log and exits with the job's outcome.

    HF_TOKEN=... python tools/gate_job.py --sha <commit> [--repo-url https://github.com/dmrai-lab/disco-space]
                                          [--flavor l4x1] [--timeout 45m] [--image python:3.11-slim]
"""
import argparse
import os
import sys
import time

SCRIPT = r'''
set -e
apt-get update -qq > /dev/null && apt-get install -y -qq git > /dev/null
git clone -q "$REPO_URL" /w && cd /w && git checkout -q "$SHA"
pip install -q pytest -r requirements.txt
python - <<'EOF'
import glob, os, site
libs = sorted(set(os.path.dirname(p) for sp in site.getsitepackages() for p in glob.glob(os.path.join(sp, "nvidia", "*", "lib"))))
open("/etc/ld.so.conf.d/nvidia-wheels.conf", "w").write("\n".join(libs) + "\n")
EOF
ldconfig
python -c "import jax; print('devices', jax.devices())"
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false pytest tests/test_acceptance.py -s -q -p no:cacheprovider
echo "=== gate.json"; cat gate.json; echo; echo "=== GATE PASSED"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True)
    ap.add_argument("--repo-url", default="https://github.com/dmrai-lab/disco-space")
    ap.add_argument("--flavor", default="l4x1")
    ap.add_argument("--timeout", default="45m")
    ap.add_argument("--image", default="python:3.11-slim")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    job = api.run_job(image=a.image, command=["bash", "-c", SCRIPT], env={"REPO_URL": a.repo_url, "SHA": a.sha},
                      flavor=a.flavor, timeout=a.timeout, name=f"disco-space-gate-{a.sha[:8]}")
    print(f"job {job.id} on {a.flavor}: {a.repo_url}@{a.sha}", flush=True)
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
