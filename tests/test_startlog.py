"""The start's progress lines (space/startlog.py): what a container says while its source loads."""
import io
import os
import time

from space import startlog

STATUS = "Name:\tpython\nVmRSS:\t 2097152 kB\nRssAnon:\t 1048576 kB\nRssFile:\t  524288 kB\nThreads:\t4\n"


def test_memory_is_read_from_the_status_text_in_bytes():
    assert startlog.memory_of(STATUS) == {"rss": 2 * 1024 ** 3, "anon": 1024 ** 3, "file": 512 * 1024 ** 2}
    assert startlog.memory_of("Name:\tx\n") == {"rss": None, "anon": None, "file": None}


def test_a_cgroup_memory_limit_is_bytes_or_none():
    assert startlog.memory_limit_from("17179869184\n") == 17179869184
    assert startlog.memory_limit_from("max\n") is None


def test_the_cache_is_summed_over_its_files_and_a_missing_cache_is_zero(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.npy").write_bytes(b"0" * 1000)
    (tmp_path / "y.npy").write_bytes(b"0" * 24)
    assert startlog.tree_bytes(str(tmp_path)) == 1024
    assert startlog.tree_bytes(str(tmp_path / "absent")) == 0


def test_the_limits_line_names_the_memory_limit_the_cpus_and_the_free_disk(tmp_path):
    m = tmp_path / "memory.max"
    m.write_text("17179869184\n")
    line = startlog.limits_line(memory_max=str(m), cpus=4, cache=str(tmp_path / "not" / "yet" / "there"))
    assert line.startswith("start: memory limit 17.2 GB, CPUs 4, free disk at the cache ") and line.endswith(" GB")
    m.write_text("max\n")
    assert "memory limit none, CPUs no quota" in startlog.limits_line(memory_max=str(m), cpus=None, cache=str(tmp_path))
    assert "memory limit unknown" in startlog.limits_line(memory_max=str(tmp_path / "absent"), cache=str(tmp_path))


def test_a_progress_line_carries_the_seconds_the_memory_and_the_cache(tmp_path):
    s = tmp_path / "status"
    s.write_text(STATUS)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "f").write_bytes(b"0" * 2_000_000)
    line = startlog.progress_line(10.0, cache=str(tmp_path / "c"), status=str(s), now=lambda: 75.0)
    assert line.startswith("start:     65 s, memory 2.1 GB (own 1.1 GB, files 0.5 GB), cache 0.0 GB, free disk ")
    absent = startlog.progress_line(10.0, status=str(tmp_path / "absent"), now=lambda: 75.0)
    assert absent == "start:     65 s, memory unknown (own unknown, files unknown)"


def test_start_prints_the_limits_then_lines_until_stop_and_the_last_message():
    out = io.StringIO()
    th = startlog.start(cache=None, cpus=2, period=0.05, out=out)
    time.sleep(0.3)
    startlog.stop("the source loaded in 3 s", out=out)
    th.join(2)
    lines = out.getvalue().splitlines()
    assert not th.is_alive()
    assert lines[0].startswith("start: memory limit ") and "CPUs 2" in lines[0]
    assert len(lines) >= 3 and all(l.startswith("start: ") for l in lines)
    assert lines[-1] == "start: the source loaded in 3 s"
    n = len(lines)
    time.sleep(0.2)
    assert len(out.getvalue().splitlines()) == n          # nothing after stop
