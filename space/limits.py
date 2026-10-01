"""The container's CPU quota, read from its cgroup, and the BLAS thread cap derived from it.

A Space's container sees the host's CPUs (192 on the ZeroGPU pool) while its cgroup grants a few; numpy's BLAS and
OpenMP pools size themselves on the visible count unless told otherwise, and XLA's CPU pool (jax, the pose expansion's
field factor) on the CPUs the process may run on, which no variable changes: measured 247 threads at 72 CPUs, 73 at
16, 25 at 4. A heavy contraction then runs on far more threads than it has cores, throttled by the quota: the pool's
first two-window warm entry spent 1217 s of 1292 s in that one jax phase against 24 s on an 8-thread box. This module
has no numpy import, so the entry can call it first."""
import math
import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def cpu_quota_from(text):
    """The CPUs a cgroup v2 ``cpu.max`` line grants: ``'<quota> <period>'`` -> ``ceil(quota / period)`` (at least 1),
    ``'max <period>'`` -> None (no quota)."""
    quota, period = text.split()[:2]
    if quota == "max":
        return None
    return max(1, math.ceil(int(quota) / int(period)))


def cpu_quota():
    """The CPUs this container's cgroup grants (v2 ``cpu.max``, else v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``),
    None when unlimited or unknown."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            return cpu_quota_from(f.read())
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as g:
            q, p = int(f.read()), int(g.read())
        return None if q < 0 else cpu_quota_from(f"{q} {p}")
    except (OSError, ValueError):
        return None


def cap_threads(quota=None):
    """Sets the BLAS / OpenMP thread variables (:data:`THREAD_VARS`) that are not already set to the cgroup's CPU
    quota (``quota`` overrides the read), before numpy is imported; returns the cap applied, None when there is no
    quota to apply. A variable the environment already sets stays as it is."""
    n = cpu_quota() if quota is None else quota
    if n is None:
        return None
    for var in THREAD_VARS:
        os.environ.setdefault(var, str(n))
    return n


def cap_affinity(quota=None):
    """Restricts this process to the first ``quota`` CPUs it may run on (``os.sched_setaffinity``; ``quota`` overrides
    the cgroup read), so that every pool that sizes itself on the schedulable CPUs (XLA's, and torch's) matches the
    quota; returns the CPUs it now runs on, or None when there is no quota or the process already runs on no more
    than it. Children (the warm-up's process, the forked GPU worker) inherit it."""
    n = cpu_quota() if quota is None else quota
    allowed = sorted(os.sched_getaffinity(0))
    if n is None or len(allowed) <= n:
        return None
    cpus = set(allowed[:n])
    os.sched_setaffinity(0, cpus)
    return cpus
