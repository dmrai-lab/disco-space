"""How the DiSCo layout reaches the container: the Hub library fetches exactly the files dmipy-replay names for the config's
classes at the pinned revision, and the layout is opened from that local directory (stand-ins; no network)."""
import json
import os

import pytest

from space import pipeline as P
from space.sources import disco as D

CFG = P.config()
MANIFEST = {"columns": {"w": "w.npy", "tiles": "tiles.npy"},
            "partitions": {"a": {"column": "p_a.npy"}, "b": {"column": "p_b.npy"}}, "groups": {}}


class Opened(Exception):
    """Raised by the stand-in for ``TimeSegmentLayout.open`` with the directory it was asked to open."""


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Stand-ins for the Hub library's two calls; ``calls`` records what the source asks for."""
    import huggingface_hub
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST))
    calls = []

    def hf_hub_download(repo, filename, **kw):
        calls.append(("file", repo, filename, kw)); return str(manifest)

    def snapshot_download(repo, **kw):
        calls.append(("snapshot", repo, kw)); return str(tmp_path / "snap")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return calls


def test_the_source_asks_the_hub_library_for_exactly_the_clients_file_list_at_the_pinned_revision(hub, tmp_path):
    d = CFG["data"]
    path = D.hub_layout_dir(d, ["a"])
    assert path == os.path.join(str(tmp_path / "snap"), d["layout"])
    assert hub[0][:3] == ("file", d["repo"], f"{d['layout']}/manifest.json") and hub[0][3]["revision"] == d["revision"]
    kind, repo, kw = hub[1]
    assert (kind, repo, kw["repo_type"], kw["revision"]) == ("snapshot", d["repo"], "dataset", d["revision"])
    assert kw["allow_patterns"] == [f"{d['layout']}/{f}" for f in ("manifest.json", "w.npy", "tiles.npy", "p_a.npy")]
    D.hub_layout_dir(d, ["a", "b"])
    assert hub[3][2]["allow_patterns"][-2:] == [f"{d['layout']}/p_a.npy", f"{d['layout']}/p_b.npy"]


def test_a_layout_on_the_hub_is_opened_from_the_downloaded_directory(hub, tmp_path, monkeypatch):
    from dmipy_replay.replay.time_segments import TimeSegmentLayout

    def opened(path, **kw):
        raise Opened(path)
    monkeypatch.setattr(TimeSegmentLayout, "open", staticmethod(opened))
    monkeypatch.delenv("DISCO_LAYOUT", raising=False)
    with pytest.raises(Opened) as e:
        D.Layout.load(CFG)
    assert e.value.args[0] == os.path.join(str(tmp_path / "snap"), CFG["data"]["layout"])
    assert [c[0] for c in hub] == ["file", "snapshot"]


@pytest.mark.parametrize("how", ["argument", "environment"])
def test_a_local_layout_downloads_nothing(hub, tmp_path, monkeypatch, how):
    from dmipy_replay.replay.time_segments import TimeSegmentLayout

    def opened(path, **kw):
        raise Opened(path)
    monkeypatch.setattr(TimeSegmentLayout, "open", staticmethod(opened))
    here = str(tmp_path / "here")
    if how == "environment":
        monkeypatch.setenv("DISCO_LAYOUT", here)
    with pytest.raises(Opened) as e:
        D.Layout.load(CFG, local=here if how == "argument" else None)
    assert e.value.args[0] == here and hub == []


def test_the_progress_line_measures_the_directory_the_hub_library_writes_to(tmp_path, monkeypatch):
    from huggingface_hub import constants
    from space import startlog
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    d = CFG["data"]
    blobs = D.hub_blobs(d)
    assert blobs == str(tmp_path / "hub" / ("datasets--" + d["repo"].replace("/", "--")) / "blobs")
    os.makedirs(blobs)
    with open(os.path.join(blobs, "abc.incomplete"), "wb") as f:
        f.write(b"0" * 3_000_000)
    assert f"cache {blobs} 0.0 GB" in startlog.progress_line(0.0, cache=blobs)
    assert startlog.tree_bytes(blobs) == 3_000_000
