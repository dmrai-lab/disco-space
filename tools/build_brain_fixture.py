"""The development fixture of the brain Space: the B.A.T.M.A.N. tutorial subject written in the layout of the brain
asset (``manifest.json``, ``fod_wm.npy``, ``fractions.npy``, ``mask.npy``, ``labels.npy``, ``regions.json``,
``stop_mask.npy``, ``mean_b0.npy``), so the ``Brain`` source is developed and tested against the asset's contract
before the MASiVar asset exists. It stays local (BATMAN's licence, dmipy-sim#193) and is for tests only.

What each file is:

* ``fod_wm.npy`` -- ``wmfod_norm.mif`` (MRtrix3 multi-tissue CSD, lmax 8, the MRtrix3 basis, which is dmipy-sim's
  required basis and dmipy-fit's) turned from the scanner frame into the image frame by ``R.T`` (``R`` from
  :meth:`dmipy_sim.phantom.Grid.from_oblique_affine`), zero outside the brain mask.
* ``fractions.npy`` -- WM / GM / CSF from ``5tt_coreg.mif`` (on the 1 mm T1 grid), averaged into every 2.5 mm
  voxel on a 3^3 sub-grid as ``examples/rph/brain_from_csd.py`` does: GM is cortical + sub-cortical GM, WM where
  the FOD has no positive l = 0 coefficient is counted as GM (no FOD, no oriented WM), zero outside the mask.
* ``labels.npy`` / ``regions.json`` -- there is no parcellation of this subject, so the GM voxels (GM fraction
  above one half) are split into 84 blocks by k-means on their coordinates (42 per hemisphere, a fixed seed),
  named ``fixture <hemisphere> <k>`` and given to 7 lobes by position: a fixture of the 84-node layout, no anatomy.
* ``stop_mask.npy`` -- the voxels whose 5TT WM fraction is above one half: where a streamline may continue.
* ``mean_b0.npy`` -- ``mean_b0_preprocessed.mif``.
* ``manifest.json`` -- the grid (the image's affine, voxel -> scanner RAS mm), the tutorial's gradient table
  (scanner frame) with the capstone's pulse timing (TE 100 / delta 25 / Delta 55 ms, which the 100 ms packs play).

    python tools/build_brain_fixture.py [--batman ~/dmrai-ws/data/batman] [--out ~/dmrai-ws/data/batman/brain_fixture]
"""
import argparse
import json
import os
import subprocess

import numpy as np

GM_COLS, WM_COL, CSF_COL = (0, 1), 2, 3                  # the 5TT columns: cortical GM, sub-cortical GM, WM, CSF
LOBES = ("frontal", "insula", "cingulate", "temporal", "parietal", "occipital", "subcortical")
TIMING = dict(TE_s=0.100, delta_s=0.025, Delta_s=0.055)  # the BATMAN capstone's (dmipy-sim#193)
SEED = 20260930


def fractions_on(shape, target_affine, tt, sub=3):
    """The five 5TT fractions averaged into every voxel of the target grid: ``sub^3`` points per voxel mapped
    through both affines and sampled trilinearly on the T1 grid; ``(X, Y, Z, 5)``."""
    from scipy.ndimage import map_coordinates
    off = (np.arange(sub) + 0.5) / sub - 0.5
    o = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    pts = (ijk[:, None, :] + o[None, :, :]).reshape(-1, 3)
    xyz = pts @ target_affine[:3, :3].T + target_affine[:3, 3]
    inv = np.linalg.inv(tt.affine)
    src = xyz @ inv[:3, :3].T + inv[:3, 3]
    out = np.stack([map_coordinates(tt.data[..., t], src.T, order=1, mode="constant", cval=0.0) for t in range(5)], -1)
    return out.reshape(tuple(shape) + (sub ** 3, 5)).mean(axis=3)


def parcellate(gm, shape):
    """84 k-means blocks of the GM voxels (42 per hemisphere, split at the median x of the brain), ``(labels,
    regions)``: labels 1..42 left, 43..84 right; per hemisphere the six blocks nearest the brain's centre are
    'subcortical' and the other 36, sorted front to back (image y), fill the six cortical lobes six at a time."""
    from scipy.cluster.vq import kmeans2
    ijk = np.argwhere(gm).astype(np.float64)
    centre = ijk.mean(0)
    labels = np.zeros(shape, np.int16)
    regions = []
    for h, (hemi, side) in enumerate((("left", ijk[:, 0] < centre[0]), ("right", ijk[:, 0] >= centre[0]))):
        pts = ijk[side]
        cents, lab = kmeans2(pts, 42, seed=SEED + h, minit="++", iter=50)
        order = np.argsort(np.linalg.norm(cents - centre, axis=1))
        sub = list(order[:6])
        cortical = sorted(order[6:], key=lambda k: -cents[k, 1])            # anterior (large y) first
        lobe_of = {k: "subcortical" for k in sub}
        for i, k in enumerate(cortical):
            lobe_of[k] = LOBES[i // 6]
        for rank, k in enumerate(sorted(range(42), key=lambda k: (LOBES.index(lobe_of[k]), -cents[k, 1]))):
            rid = 1 + 42 * h + rank
            v = pts[lab == k].astype(int)
            labels[v[:, 0], v[:, 1], v[:, 2]] = rid
            regions.append(dict(id=rid, name=f"fixture {hemi} {rank + 1:02d}", hemisphere=hemi, lobe=lobe_of[k]))
    return labels, regions


def main(argv=None):
    from dmipy_sim.io.mrtrix import read_mif
    from dmipy_sim.phantom import Grid
    from dmipy_sim.replay.so3 import rotate_sh
    ap = argparse.ArgumentParser()
    ap.add_argument("--batman", default=os.path.expanduser("~/dmrai-ws/data/batman"))
    ap.add_argument("--out", default=os.path.expanduser("~/dmrai-ws/data/batman/brain_fixture"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    fod = read_mif(os.path.join(a.batman, "wmfod_norm.mif"))
    mask = np.asarray(read_mif(os.path.join(a.batman, "mask_den_unr_preproc_unb.mif")).data, bool)
    tt = read_mif(os.path.join(a.batman, "5tt_coreg.mif"))
    b0 = read_mif(os.path.join(a.batman, "mean_b0_preprocessed.mif"))
    shape = fod.shape[:3]
    _, R = Grid.from_oblique_affine(fod.affine, shape)
    sh = rotate_sh(np.asarray(fod.data, np.float64), np.asarray(R).T) * mask[..., None]
    F = fractions_on(shape, fod.affine, tt) * mask[..., None]
    has_fod = sh[..., 0] > 0
    f_wm = F[..., WM_COL] * has_fod
    f_gm = F[..., GM_COLS[0]] + F[..., GM_COLS[1]] + F[..., WM_COL] * ~has_fod
    f_csf = F[..., CSF_COL]
    fractions = np.stack([f_wm, f_gm, f_csf], -1)
    labels, regions = parcellate(mask & (f_gm > 0.5), shape)
    stop = mask & (F[..., WM_COL] > 0.5)
    g = np.loadtxt(os.path.join(a.batman, "dwipreproc_grad.b"))
    commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    groups = []
    for r in regions:
        name = f"{r['hemisphere'][0].upper()} {r['lobe']}"
        grp = next((x for x in groups if x["name"] == name), None)
        if grp is None:
            grp = dict(name=name, hemisphere=r["hemisphere"], lobe=r["lobe"], ids=[]); groups.append(grp)
        grp["ids"].append(r["id"])
    manifest = {
        "subject": "BATMAN (fixture)",
        "source": {"dataset": "B.A.T.M.A.N. MRtrix3 tutorial (Tahedl 2018), Supplementary_Files", "doi": "10.17605/OSF.IO/FKYHT",
                   "paper_doi": None, "license": "local development fixture only (dmipy-sim#193 item 1)"},
        "grid": {"shape": list(shape), "voxel_size_mm": [float(x) for x in np.linalg.norm(fod.affine[:3, :3], axis=0)],
                 "affine": np.asarray(fod.affine, float).tolist()},
        "protocol": {"bvals_s_mm2": [float(round(b)) for b in g[:, 3]], "bvecs": g[:, :3].tolist(), "bvec_frame": "scanner", **TIMING},
        "fod": {"basis": "tournier07", "lmax": 8, "frame": "image"},
        "fractions": ["wm", "gm", "csf"],
        "reconstruction": {"by": "MRtrix3 dwi2fod msmt_csd (the tutorial's wmfod_norm.mif), 5TT fractions", "fixture": True},
        "parcellation": {"method": "fixture: k-means blocks of the GM voxels, 42 per hemisphere, seed %d" % SEED, "n_regions": len(regions)},
        "built_by": {"tool": "disco-space tools/build_brain_fixture.py", "commit": commit},
    }
    np.save(os.path.join(a.out, "fod_wm.npy"), sh.astype(np.float32))
    np.save(os.path.join(a.out, "fractions.npy"), fractions.astype(np.float16))
    np.save(os.path.join(a.out, "mask.npy"), mask)
    np.save(os.path.join(a.out, "labels.npy"), labels)
    np.save(os.path.join(a.out, "stop_mask.npy"), stop)
    np.save(os.path.join(a.out, "mean_b0.npy"), np.asarray(b0.data, np.float32))
    with open(os.path.join(a.out, "regions.json"), "w") as f:
        json.dump(dict(regions=regions, groups=groups), f, indent=1)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{a.out}: grid {shape}, {int(mask.sum())} brain voxels, {int((f_wm > 0).sum())} with WM, {int(stop.sum())} in the stop mask, "
          f"{len(regions)} regions ({len(groups)} lobar groups), {len(g)} measurements")


if __name__ == "__main__":
    main()
