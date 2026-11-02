"""The acceptance gate as a Hugging Face Job in the Space's own image (``hf.co/spaces/<space>``: the image Hugging
Face built from the deployed Dockerfile, tests included), so the gate runs on exactly what is deployed and installs
nothing but pytest. The job runs ``tests/test_acceptance.py`` and prints ``gate.json``; this script starts it, waits,
prints its log and exits with the job's outcome. The Space is private, so the container gets the caller's Hub token
as the job secret ``HF_TOKEN`` (the layout download reads it too).

The job starts only once the Space runs the deployed commit (:func:`wait_for_build`): a deploy that changes a variable
rebuilds the image, and a job started on an image still building errors with no log.

    python tools/gate_job.py [--space rfick/disco] [--flavor l4x1] [--timeout 45m] [--build-timeout 2400]
"""
import argparse
import os
import sys
import time

POLL_S = 30


def wait_for_build(api, space, rev, timeout_s):
    """Return once ``space`` runs commit ``rev`` (stage RUNNING with that sha, on two polls ``POLL_S`` apart, so a stage
    read just before a rebuild starts does not count); exit on an error stage or after ``timeout_s``."""
    t0, seen = time.time(), 0
    time.sleep(POLL_S)
    while True:
        r = api.get_space_runtime(space)
        stage, sha = r.stage, (r.raw or {}).get("sha")
        if str(stage).endswith("ERROR"):
            sys.exit(f"the Space is in {stage} at {sha}: the image to test did not build or start")
        seen = seen + 1 if (stage == "RUNNING" and sha == rev) else 0
        if seen >= 2:
            print(f"{space} runs {rev[:12]} ({time.time() - t0:.0f} s after the deploy)", flush=True)
            return
        if time.time() - t0 > timeout_s:
            sys.exit(f"the Space is still {stage} at {sha} after {timeout_s} s, not RUNNING at {rev}")
        time.sleep(POLL_S)

SCRIPT = r'''
set -e
cd /app
pip install -q --retries 10 pytest
python -c "import jax; print('devices', jax.devices())"
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true HF_HOME=/tmp/hf DMIPY_REPLAY_CACHE=/tmp/replay JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache \
  GATE_OUT=/tmp/gate.json PYTHONPATH=/app pytest tests/test_acceptance.py -s -q -p no:cacheprovider
echo "=== gate.json"; cat /tmp/gate.json; echo; echo "=== GATE PASSED"
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=os.environ.get("DISCO_SPACE", "rfick/disco"))
    ap.add_argument("--flavor", default="l4x1")
    ap.add_argument("--timeout", default="45m")
    ap.add_argument("--build-timeout", type=int, default=2400, help="seconds to wait for the Space to run the deployed commit")
    a = ap.parse_args()
    from huggingface_hub import HfApi, get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)
    rev = api.repo_info(a.space, repo_type="space").sha
    wait_for_build(api, a.space, rev, a.build_timeout)
    job = api.run_job(image=f"hf.co/spaces/{a.space}", command=["bash", "-c", SCRIPT],
                      secrets={"HF_TOKEN": token}, flavor=a.flavor, timeout=a.timeout, name=f"disco-space-gate-{rev[:8]}")
    print(f"job {job.id} on {a.flavor}: the image of {a.space} (repo at {rev})", flush=True)
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
