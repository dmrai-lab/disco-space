"""Desikan-Killiany + aseg (the MRtrix ``fs_default`` convention, 84 nodes: 34 cortical + 8 subcortical per
hemisphere), majority-voted from a SynthSeg segmentation in T1 space onto a diffusion grid, plus a 14-node lobar
grouping (7 lobes x 2 hemispheres) and a white-matter stop mask for tracking.

The segmentation itself is made *outside* this repo by ``SynthSeg --parc --robust`` (BBillot/SynthSeg;
https://github.com/BBillot/SynthSeg) on the T1, run in its own environment (this script has no TensorFlow
dependency). This module only consumes that NIfTI: it registers the T1 onto the diffusion grid (rigid, dipy, mutual
information) and resamples the labels with true majority voting -- for every diffusion voxel, every label found
among its ``factor**3`` sub-points (sampled at the T1's native resolution through the fitted transform) is counted
and the plurality wins, rather than a single nearest-neighbour point sample.

    python tools/build_parcellation.py --t1 T1w.nii.gz --synthseg aseg.nii.gz --ref mean_b0.nii.gz --out-dir OUT

Writes, on the reference (diffusion) grid, under ``--out-dir``:
    labels.npy      int16 (X, Y, Z): compact node id 1..84 in the 84 regions, 0 elsewhere
    stop_mask.npy   bool  (X, Y, Z): True where the segmentation's majority label is white matter (a streamline may
                    continue there; leaving white matter into cortex, a ventricle or CSF ends it)
    regions.json    the 84 nodes (id, name, hemisphere, lobe) and the 14 lobar groups
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# ---- the FreeSurfer label tables (FreeSurferColorLUT.txt), restricted to what SynthSeg --parc --robust emits -----

# The 34 Desikan-Killiany cortical regions, FreeSurfer's own (alphabetical) ctx-lh-* numbering, 1001..1035 skipping
# 1004 (corpuscallosum, not part of the DK pial surface labelling and not emitted by SynthSeg); ctx-rh-* is +1000.
CORTICAL_NAMES = {
    1001: "bankssts", 1002: "caudalanteriorcingulate", 1003: "caudalmiddlefrontal", 1005: "cuneus",
    1006: "entorhinal", 1007: "fusiform", 1008: "inferiorparietal", 1009: "inferiortemporal",
    1010: "isthmuscingulate", 1011: "lateraloccipital", 1012: "lateralorbitofrontal", 1013: "lingual",
    1014: "medialorbitofrontal", 1015: "middletemporal", 1016: "parahippocampal", 1017: "paracentral",
    1018: "parsopercularis", 1019: "parsorbitalis", 1020: "parstriangularis", 1021: "pericalcarine",
    1022: "postcentral", 1023: "posteriorcingulate", 1024: "precentral", 1025: "precuneus",
    1026: "rostralanteriorcingulate", 1027: "rostralmiddlefrontal", 1028: "superiorfrontal",
    1029: "superiorparietal", 1030: "superiortemporal", 1031: "supramarginal", 1032: "frontalpole",
    1033: "temporalpole", 1034: "transversetemporal", 1035: "insula",
}

# The 8 subcortical structures per hemisphere SynthSeg --parc --robust emits (aseg ids); left and right are not a
# uniform offset of one another, so both are given explicitly.
SUBCORTICAL_NAMES_LEFT = {
    10: "Thalamus", 11: "Caudate", 12: "Putamen", 13: "Pallidum",
    17: "Hippocampus", 18: "Amygdala", 26: "Accumbens-area", 28: "VentralDC",
}
SUBCORTICAL_NAMES_RIGHT = {
    49: "Thalamus", 50: "Caudate", 51: "Putamen", 52: "Pallidum",
    53: "Hippocampus", 54: "Amygdala", 58: "Accumbens-area", 60: "VentralDC",
}

# A standard six-lobe grouping of the 34 DK cortical regions (Klein & Tourville 2012-style), plus a seventh
# "subcortical" group per hemisphere for the 8 subcortical structures: 7 lobes x 2 hemispheres = 14 groups.
LOBE_OF_CORTICAL = {
    "superiorfrontal": "frontal", "rostralmiddlefrontal": "frontal", "caudalmiddlefrontal": "frontal",
    "parsopercularis": "frontal", "parstriangularis": "frontal", "parsorbitalis": "frontal",
    "lateralorbitofrontal": "frontal", "medialorbitofrontal": "frontal", "precentral": "frontal",
    "paracentral": "frontal", "frontalpole": "frontal",
    "superiorparietal": "parietal", "inferiorparietal": "parietal", "supramarginal": "parietal",
    "postcentral": "parietal", "precuneus": "parietal",
    "superiortemporal": "temporal", "middletemporal": "temporal", "inferiortemporal": "temporal",
    "bankssts": "temporal", "fusiform": "temporal", "transversetemporal": "temporal", "entorhinal": "temporal",
    "temporalpole": "temporal", "parahippocampal": "temporal",
    "lateraloccipital": "occipital", "lingual": "occipital", "cuneus": "occipital", "pericalcarine": "occipital",
    "caudalanteriorcingulate": "cingulate", "rostralanteriorcingulate": "cingulate",
    "posteriorcingulate": "cingulate", "isthmuscingulate": "cingulate",
    "insula": "insula",
}
assert set(LOBE_OF_CORTICAL) == set(CORTICAL_NAMES.values())
LOBES = ("frontal", "parietal", "temporal", "occipital", "cingulate", "insula", "subcortical")

# White-matter labels: a streamline may continue here (cerebral + cerebellar white matter, brain-stem); leaving
# this set (into cortex, a ventricle or CSF) ends it. Judgement call, not part of the 84-node parcellation itself.
WM_STOP_LABELS = frozenset({2, 41, 7, 46, 16})   # Left/Right-Cerebral-WM, Left/Right-Cerebellum-WM, Brain-Stem


def node_table():
    """The 84 nodes in a fixed, documented order: node id 1..42 is the left hemisphere (34 cortical by ascending
    FreeSurfer id, then the 8 subcortical structures), 43..84 the same for the right. Each entry is
    ``{"id", "fs_label", "name", "hemisphere", "lobe"}``."""
    nodes = []
    for hemi, cortical, subcortical in (
        ("left", CORTICAL_NAMES, SUBCORTICAL_NAMES_LEFT),
        ("right", {k + 1000: v for k, v in CORTICAL_NAMES.items()}, SUBCORTICAL_NAMES_RIGHT),
    ):
        for fs_id in sorted(cortical):
            name = cortical[fs_id]
            nodes.append({"fs_label": fs_id, "name": name, "hemisphere": hemi, "lobe": LOBE_OF_CORTICAL[name]})
        for fs_id in sorted(subcortical):
            nodes.append({"fs_label": fs_id, "name": subcortical[fs_id], "hemisphere": hemi, "lobe": "subcortical"})
    for i, node in enumerate(nodes, start=1):
        node["id"] = i
    assert len(nodes) == 84
    return nodes


def regions_payload():
    """``regions.json``'s content: the 84 nodes plus the 14 lobar groups (lobe x hemisphere)."""
    nodes = node_table()
    groups = [f"{lobe}-{hemi}" for lobe in LOBES for hemi in ("left", "right")]
    return {
        "convention": "Desikan-Killiany + aseg (MRtrix fs_default, 84 nodes: 34 cortical + 8 subcortical per hemisphere)",
        "regions": nodes,
        "lobes": groups,
    }


def register_t1_to_ref(t1_data, t1_affine, ref_data, ref_affine):
    """The rigid (6 DOF) transform of ``t1``'s world space onto ``ref``'s (dipy, mutual information: a standard
    cross-modal registration). Returns the ``AffineMap``-style 4x4 matrix mapping a point in ``ref``'s world space
    to the corresponding point in ``t1``'s world space (dipy's convention: domain -> codomain)."""
    from dipy.align.imaffine import transform_centers_of_mass, AffineRegistration, MutualInformationMetric
    from dipy.align.transforms import RigidTransform3D

    com = transform_centers_of_mass(ref_data, ref_affine, t1_data, t1_affine)
    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.3)
    affreg = AffineRegistration(metric=metric, level_iters=[1000, 200, 50], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1])
    rigid = affreg.optimize(ref_data, t1_data, RigidTransform3D(), params0=None,
                             static_grid2world=ref_affine, moving_grid2world=t1_affine,
                             starting_affine=com.affine)
    return rigid.affine


def _supersampled_affine(ref_affine, factor):
    """The grid-to-world affine of ``ref``'s grid split ``factor`` ways per axis, each sub-voxel centred within its
    parent voxel (sub-index ``k``'s continuous parent coordinate is ``(k + 0.5) / factor - 0.5``)."""
    c = 0.5 / factor - 0.5
    m = np.eye(4)
    m[0, 0] = m[1, 1] = m[2, 2] = 1.0 / factor
    m[:3, 3] = c
    return ref_affine @ m


def majority_vote(seg_data, seg_affine, ref_shape, ref_affine, t1_to_ref_affine, factor=4):
    """The segmentation's majority label per voxel of ``ref``'s grid: every voxel is split into ``factor**3``
    sub-points (sampled through the fitted rigid transform at nearest-neighbour), and the plurality label among them
    is kept. ``0`` where nothing in ``seg_data`` maps there. Returns an ``int32`` array of shape ``ref_shape``."""
    from dipy.align.imaffine import AffineMap

    if factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor}")
    sup_shape = tuple(int(s) * factor for s in ref_shape)
    sup_affine = _supersampled_affine(ref_affine, factor)
    amap = AffineMap(t1_to_ref_affine, domain_grid_shape=sup_shape, domain_grid2world=sup_affine,
                      codomain_grid_shape=seg_data.shape, codomain_grid2world=seg_affine)
    sup_labels = amap.transform(seg_data.astype(np.int32), interpolation="nearest")

    X, Y, Z = ref_shape
    blocks = sup_labels.reshape(X, factor, Y, factor, Z, factor)
    blocks = np.transpose(blocks, (0, 2, 4, 1, 3, 5)).reshape(X, Y, Z, factor ** 3)

    uniq, inverse = np.unique(blocks, return_inverse=True)
    inverse = inverse.reshape(X * Y * Z, factor ** 3)
    counts = np.zeros((X * Y * Z, len(uniq)), dtype=np.int32)
    for k in range(factor ** 3):
        np.add.at(counts, (np.arange(X * Y * Z), inverse[:, k]), 1)
    majority = uniq[counts.argmax(axis=1)].reshape(X, Y, Z)
    return majority.astype(np.int32)


def labels_and_stop_mask(majority_fs_labels):
    """``(labels, stop_mask)`` on ``majority_fs_labels``'s grid: ``labels`` is the compact node id (1..84, 0
    elsewhere) of :func:`node_table`; ``stop_mask`` is True on :data:`WM_STOP_LABELS`."""
    nodes = node_table()
    fs_to_node = {n["fs_label"]: n["id"] for n in nodes}
    labels = np.zeros(majority_fs_labels.shape, dtype=np.int16)
    for fs_id, node_id in fs_to_node.items():
        labels[majority_fs_labels == fs_id] = node_id
    stop_mask = np.isin(majority_fs_labels, list(WM_STOP_LABELS))
    return labels, stop_mask


def label_volumes_mm3(labels, voxel_volume_mm3):
    """``{node_id: volume_mm3}`` for every node present in ``labels`` (0 excluded)."""
    vals, counts = np.unique(labels, return_counts=True)
    return {int(v): float(c) * voxel_volume_mm3 for v, c in zip(vals, counts) if v != 0}


def build_parcellation(t1_path, synthseg_path, ref_path, out_dir, factor=4):
    """The CLI's body: register, majority-vote, write ``labels.npy``, ``stop_mask.npy``, ``regions.json`` under
    ``out_dir``. Returns ``(labels, stop_mask, regions)`` for a caller (``build_brain_asset.py``) that wants them
    in memory instead of round-tripping through disk."""
    import os
    import nibabel as nib

    t1_im = nib.load(t1_path)
    seg_im = nib.load(synthseg_path)
    ref_im = nib.load(ref_path)
    if seg_im.shape != t1_im.shape or not np.allclose(seg_im.affine, t1_im.affine, atol=1e-3):
        raise ValueError("the SynthSeg output must be on the T1's own grid (it is by construction unless T1 was "
                          "not already at the resolution SynthSeg resamples to internally)")

    t1_data = np.asarray(t1_im.dataobj, dtype=np.float64)
    ref_data = np.asarray(ref_im.dataobj, dtype=np.float64)
    seg_data = np.asarray(seg_im.dataobj, dtype=np.int32)

    t1_to_ref = register_t1_to_ref(t1_data, t1_im.affine, ref_data, ref_im.affine)
    majority_fs = majority_vote(seg_data, seg_im.affine, ref_im.shape, ref_im.affine, t1_to_ref, factor=factor)
    labels, stop_mask = labels_and_stop_mask(majority_fs)
    regions = regions_payload()

    voxel_mm3 = float(np.abs(np.linalg.det(ref_im.affine[:3, :3])))
    volumes = label_volumes_mm3(labels, voxel_mm3)
    n_present = sum(1 for n in regions["regions"] if n["id"] in volumes)
    print(f"{n_present}/84 nodes present on the diffusion grid")
    if volumes:
        v = np.array(list(volumes.values()))
        print(f"per-label volume mm3: min {v.min():.1f}, median {np.median(v):.1f}, max {v.max():.1f}")

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "stop_mask.npy"), stop_mask)
    with open(os.path.join(out_dir, "regions.json"), "w") as f:
        json.dump(regions, f, indent=2)
    return labels, stop_mask, regions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t1", required=True, help="the subject's T1 (the grid SynthSeg's output is on)")
    ap.add_argument("--synthseg", required=True, help="SynthSeg --parc --robust's output NIfTI, on the T1 grid")
    ap.add_argument("--ref", required=True, help="a NIfTI defining the diffusion grid (e.g. the mean b=0)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--factor", type=int, default=4, help="sub-points per axis per diffusion voxel for the "
                     "majority vote (default 4, i.e. 64 sub-points)")
    a = ap.parse_args()
    build_parcellation(a.t1, a.synthseg, a.ref, a.out_dir, factor=a.factor)


if __name__ == "__main__":
    main()
