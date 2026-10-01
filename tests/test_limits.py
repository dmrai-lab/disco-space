"""space.limits: the cgroup CPU quota and the thread cap derived from it."""
import os

import pytest

from space import limits as L


def test_the_quota_is_the_ceiling_of_quota_over_period():
    assert L.cpu_quota_from("100000 100000") == 1
    assert L.cpu_quota_from("150000 100000") == 2
    assert L.cpu_quota_from("800000 100000\n") == 8
    assert L.cpu_quota_from("50000 100000") == 1
    assert L.cpu_quota_from("max 100000") is None
    with pytest.raises(ValueError):
        L.cpu_quota_from("")


def test_the_cap_sets_only_the_unset_variables(monkeypatch):
    for var in L.THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    assert L.cap_threads(quota=6) == 6
    assert os.environ["OMP_NUM_THREADS"] == "6" and os.environ["OPENBLAS_NUM_THREADS"] == "6"
    assert os.environ["MKL_NUM_THREADS"] == "3"
