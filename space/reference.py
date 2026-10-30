"""The acceptance gate's reference: the replay pack the layout was contracted from, replayed walker by walker at a set
of voxels (only those voxels' rows are read, by range, from the Hub or a local copy). This is the full replay, not a
rendering: every stored band, every field mode, the contact, the pools; the layout's image of the same rows must equal
it to ``[gate] reference_max_abs_diff`` (S/S0)."""
from __future__ import annotations

import numpy as np


def open_pack(uri, workers=8):
    """The columnar replay pack at ``uri`` (``hf://owner/name/prefix`` or a directory)."""
    from dmipy_sim.replay.columnar import ColumnarPack
    return ColumnarPack(uri, workers=workers)


def gate_voxels(pack, n):
    """``n`` voxels spread evenly over the pack's index (its (voxel, pool) rows in order), as ijk triples: the spot
    check's rule."""
    keys = sorted({tuple(r["ijk"]) for r in pack.index["rows"]})
    pick = np.unique(np.linspace(0, len(keys) - 1, min(int(n), len(keys))).round().astype(int))
    return [keys[i] for i in pick]


def replay_at(pack, voxels, sequence, tissue=None, scanner=None):
    """``(n_voxels, n_meas)``: each voxel's weighted ensemble magnitude under ``sequence`` (normalised by its walkers'
    weight, as the layout's image is), from the pack's own rows at ``voxels`` (every band; the field modes and the
    contact read when the tissue needs them)."""
    from dmipy_sim.phantom.grid import Grid
    grid = Grid.from_meta(pack.index["grid"])
    need_field = tissue is not None and (getattr(tissue, "chi_iso", None) or getattr(tissue, "chi_aniso", None))
    view = pack.view(voxels=voxels, contact=tissue is not None,
                     modes=(pack.path_groups[-1] if need_field and pack.path_groups else 0))
    w, ew, E = view.walker_signals(sequence, tissue=tissue, scanner=scanner, tol=None)
    ijk, _ = grid.bin(view.r0)
    v = np.ravel_multi_index(ijk.T, tuple(grid.shape))
    flat = np.array([np.ravel_multi_index(k, tuple(grid.shape)) for k in voxels])
    pos = {int(f): i for i, f in enumerate(flat)}
    idx = np.array([pos[int(x)] for x in v])
    num = np.zeros((len(voxels), E.shape[1]), complex)
    np.add.at(num, idx, np.asarray(ew)[:, None] * np.asarray(E))
    den = np.bincount(idx, np.asarray(w, np.float64), minlength=len(voxels))
    return np.abs(num / den[:, None])
