"""What a starting container is doing, said once every few seconds: the source loads before the page serves (the
Hub download, then the layout's host arrays), and a start that says nothing cannot be told from one that hangs.

:func:`start` prints the container's limits once (memory and CPUs from its cgroup, the free disk where the cache directory
lives) and then, until :func:`stop`, one line per period: the seconds since the start, this process's resident
memory (anonymous and file-backed), the bytes in the cache directory (named in the line) and the free disk. Lines are flushed. No numpy import."""
import os
import shutil
import sys
import threading
import time

PERIOD_S = 30.0


def memory_of(status_text):
    """``/proc/<pid>/status`` -> ``{"rss", "anon", "file"}`` in bytes; a field the kernel does not give is None."""
    kb = {}
    for line in status_text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmRSS", "RssAnon", "RssFile"):
            kb[key] = 1024 * int(rest.split()[0])
    return {name: (kb[key] if key in kb else None) for name, key in (("rss", "VmRSS"), ("anon", "RssAnon"), ("file", "RssFile"))}


def memory_limit_from(text):
    """A cgroup v2 ``memory.max`` value in bytes, None for ``max`` (no limit)."""
    text = text.strip()
    return None if text == "max" else int(text)


def tree_bytes(path):
    """The bytes of the regular files under ``path`` (0 when it does not exist); a file that vanishes while the tree
    is walked (a download's temporary file renamed into place) is skipped."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _gb(n):
    return "unknown" if n is None else f"{n / 1e9:.1f} GB"


def _free_disk(cache):
    """The free bytes of the filesystem that holds ``cache`` (its nearest existing parent), None when unknown."""
    probe = cache
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        probe = None if parent == probe else parent
    return shutil.disk_usage(probe).free if probe else None


def limits_line(*, memory_max="/sys/fs/cgroup/memory.max", cpus=None, cache=None):
    """The one line of a container's limits: its cgroup's memory, the CPUs it grants, the free disk at the cache."""
    try:
        with open(memory_max) as f:
            limit = memory_limit_from(f.read())
        mem = "none" if limit is None else _gb(limit)
    except (OSError, ValueError):
        mem = "unknown"
    return f"start: memory limit {mem}, CPUs {cpus if cpus else 'no quota'}, free disk at the cache {_gb(_free_disk(cache)) if cache else 'unknown'}"


def progress_line(t0, *, cache=None, status="/proc/self/status", now=time.monotonic):
    """One progress line: seconds since ``t0``, this process's memory, the bytes under the cache directory ``cache`` (named in the line) and the free disk."""
    try:
        with open(status) as f:
            m = memory_of(f.read())
    except OSError:
        m = {"rss": None, "anon": None, "file": None}
    line = f"start: {now() - t0:6.0f} s, memory {_gb(m['rss'])} (own {_gb(m['anon'])}, files {_gb(m['file'])})"
    if cache:
        line += f", cache {cache} {tree_bytes(cache) / 1e9:.1f} GB, free disk {_gb(_free_disk(cache))}"
    return line


_stop = threading.Event()


def start(*, cache=None, cpus=None, period=PERIOD_S, out=None):
    """Print the limits now and a progress line every ``period`` seconds until :func:`stop`; returns the thread."""
    out = out or sys.stdout
    t0 = time.monotonic()
    _stop.clear()
    print(limits_line(cpus=cpus, cache=cache), file=out, flush=True)

    def loop():
        while not _stop.wait(period):
            print(progress_line(t0, cache=cache), file=out, flush=True)
    th = threading.Thread(target=loop, name="startlog", daemon=True)
    th.start()
    return th


def stop(message=None, *, out=None):
    """End the progress lines; ``message`` is printed as the start's last line."""
    _stop.set()
    if message:
        print(f"start: {message}", file=out or sys.stdout, flush=True)
