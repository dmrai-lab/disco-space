"""The brain Space's asset (disco-space#8 step 4): a MASiVar scan's multi-shell DWI, reconstructed by dmipy-fit
*alone* (three-tissue responses, Dhollander 2016; multi-shell multi-tissue CSD, Jeurissen 2014, ``CsdCvxpyOptimizer``)
into a WM FOD field and WM/GM/CSF fractions, plus the parcellation of ``build_parcellation.py``, written to an asset
directory the Space's ``Brain`` source reads. The FOD and the fractions are dmipy-fit's
``MultiCompartmentSphericalHarmonicsModel`` end to end; MRtrix appears nowhere. dipy is used by this offline tool for
what is not reconstruction: the median-Otsu brain mask and the peak count of the sanity check (and inside dmipy-fit's
own tissue-response estimator until dmipy-fit#39 replaces it). Nothing here runs in the Space.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/build_brain_asset.py --dwi-dir DIR --parcellation-dir PARC_DIR \\
        --out-dir OUT [--n-test-voxels 200] [--solver csd_cvxpy]

``--dwi-dir`` holds the PreQual-preprocessed single-shell NIfTI + bval + bvec triples of one scan (the derivative
keeps every shell separate; this script is what merges them into one multi-shell volume). ``--parcellation-dir`` is
``build_parcellation.py``'s output (``labels.npy``, ``stop_mask.npy``, ``regions.json``); it is resampled here onto
the DWI's own grid only if its own grid differs (it does not, by construction: both scripts are handed the same
mean b=0 as the reference grid).

With ``--n-test-voxels`` set (no ``--out-dir`` write of the whole-brain arrays), fits only a random subset of masked
voxels and reports the measured per-voxel time and the whole-brain extrapolation, for a dry run before the (long)
full fit.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np

SH_ORDER = 8
B0_THRESHOLD_S_MM2 = 50.0                # a row below this b-value (s/mm^2) is a b = 0 measurement


def merge_shells(dwi_dir):
    """The PreQual per-shell NIfTI + bval + bvec triples of ``dwi_dir`` (one file per shell, each already carrying
    its own b = 0 volume), concatenated into one multi-shell acquisition: ``(data, bvals, bvecs, affine, shells)``.
    ``data`` is ``(X, Y, Z, N)`` float32, ``bvals`` (s/mm^2) and ``bvecs`` (image-frame unit vectors, BIDS/FSL
    convention) are ``(N,)`` / ``(N, 3)``; ``shells`` lists ``(b_value, n_directions)`` as found, ascending."""
    import nibabel as nib

    niis = sorted(glob.glob(os.path.join(dwi_dir, "*_dwi.nii.gz")))
    if not niis:
        raise FileNotFoundError(f"no *_dwi.nii.gz in {dwi_dir}")
    datas, bvals, bvecs, shells, affine, ref_shape = [], [], [], [], None, None
    for f in niis:
        stem = f[: -len(".nii.gz")]
        im = nib.load(f)
        if affine is None:
            affine, ref_shape = im.affine, im.shape[:3]
        elif not np.allclose(im.affine, affine, atol=1e-4) or im.shape[:3] != ref_shape:
            raise ValueError(f"{f} is not on the other shells' grid")
        d = np.asarray(im.dataobj, dtype=np.float32)
        bv = np.loadtxt(stem + ".bval")
        vec = np.loadtxt(stem + ".bvec").T
        datas.append(d)
        bvals.append(bv)
        bvecs.append(vec)
        n_dwi = int((bv >= B0_THRESHOLD_S_MM2).sum())
        if n_dwi:
            shells.append((float(np.median(bv[bv >= B0_THRESHOLD_S_MM2])), n_dwi))
    data = np.concatenate(datas, axis=-1)
    bvals = np.concatenate(bvals)
    bvecs = np.concatenate(bvecs, axis=0)
    shells = sorted(shells)
    return data, bvals, bvecs, affine, shells


def mean_b0(data, bvals):
    """The mean of every b = 0 volume (``bvals < B0_THRESHOLD_S_MM2``), ``(X, Y, Z)`` float32."""
    b0 = bvals < B0_THRESHOLD_S_MM2
    if not np.any(b0):
        raise ValueError("no b = 0 measurement in this acquisition")
    return data[..., b0].mean(axis=-1).astype(np.float32)


def brain_mask(mean_b0_volume):
    """The brain mask from the mean b = 0 image (dipy's median-Otsu, the standard tool; not dmipy-fit's job)."""
    from dipy.segment.mask import median_otsu
    _, mask = median_otsu(mean_b0_volume, median_radius=4, numpass=4)
    return mask.astype(bool)


def _tissue_scheme_check_workaround(self, acquisition_scheme):
    """Replaces ``MultiCompartmentModelProperties._check_tissue_model_acquisition_scheme`` (works around a
    dmipy-fit gap, as of commit f6741c3, no issue filed yet -- flag for dmipy-fit#39): the original builds
    ``np.testing.assert_array_almost_equal([bvalues, delta, Delta, gradient_strengths], ...)`` over two schemes, but
    when timing is unmeasured (``delta``/``Delta`` unset, as in this dataset -- see the manifest's
    ``protocol.delta_s``/``Delta_s`` = null) ``shell_delta``/``shell_Delta``/``shell_gradient_strengths`` are
    ``None`` and stacking ``[array, None, None, None]`` raises ``ValueError`` in current numpy before the intended
    comparison ever runs -- on *any* scheme missing timing, matching or not. This compares field by field instead
    (``None`` equal to ``None``). A module-level function (not a closure) so the patched instance stays picklable
    (the CSD fit can be process-parallel; see :func:`fit_msmt_csd`'s docstring for why it is not, regardless)."""
    for model in self.models:
        if model._model_type != "TissueResponseModel":
            continue
        for a, b in ((acquisition_scheme.shell_bvalues, model.acquisition_scheme.shell_bvalues),
                     (acquisition_scheme.shell_delta, model.acquisition_scheme.shell_delta),
                     (acquisition_scheme.shell_Delta, model.acquisition_scheme.shell_Delta),
                     (acquisition_scheme.shell_gradient_strengths, model.acquisition_scheme.shell_gradient_strengths)):
            if a is None or b is None:
                if a is not b:
                    raise ValueError("Acquisition scheme of MC-model and tissue response model are not the same.")
            else:
                np.testing.assert_array_almost_equal(a, b)


def _patch_tissue_scheme_check(mc_model):
    import types
    mc_model._check_tissue_model_acquisition_scheme = types.MethodType(_tissue_scheme_check_workaround, mc_model)


def fit_msmt_csd(data, bvals, bvecs, mask, sh_order=SH_ORDER, solver="csd_cvxpy", voxel_positions=None):
    """dmipy-fit end to end: :func:`three_tissue_response_dhollander16` estimates the WM / GM / CSF response
    kernels from the data itself, then ``MultiCompartmentSphericalHarmonicsModel`` (``S0_tissue_responses`` set,
    volume fractions free) fits every voxel of ``mask`` with ``solver`` (``CsdCvxpyOptimizer`` for
    ``'csd_cvxpy'``). If ``voxel_positions`` is given (a ``(mask.sum(),)``-shaped boolean subset, for a dry run) only
    those voxels are fit; the rest of ``mask`` is still used to estimate the responses.

    Always fits serially (``use_parallel_processing=False``): dmipy-fit's parallel path forks a
    ``ProcessPoolExecutor``, which the JAX import this fit chain pulls in (``dmipy_sim.replay.so3``, for the SH
    basis) explicitly warns is unsafe to fork after (JAX is itself multithreaded) -- and separately, the picked-up
    ``mc`` instance carries the bound-method workaround above, which is not guaranteed picklable to a worker
    process either. Thread-level parallelism (BLAS/OSQP) is still capped by ``OMP_NUM_THREADS``/``MKL_NUM_THREADS``
    in the environment, per this lab's shared-box rule.

    Returns ``(sh, fractions, S0_responses, response_models, scheme, seconds)``: ``sh`` is ``(X, Y, Z, n_coef)`` (the
    WM FOD, zero outside the fitted voxels), ``fractions`` is ``(X, Y, Z, 3)`` (wm, gm, csf, zero outside), the rest
    are for the manifest / checks."""
    from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
    from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
    from dmipy_fit.tissue_response.three_tissue_response import three_tissue_response_dhollander16

    scheme = acquisition_scheme_from_bvalues(bvals * 1e6, bvecs, b0_threshold=B0_THRESHOLD_S_MM2 * 1e6)
    data = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0)

    t0 = time.perf_counter()
    # wm_algorithm='tournier07' (not the default 'tournier13'): tournier13's internal SH-model refit compares its
    # own re-derived acquisition scheme against the caller's and fails when delta/Delta are unknown (no timing in
    # this dataset's sidecars, see the manifest's protocol.delta_s/Delta_s = null); tournier07 is FA-based and
    # avoids that path. It is also what the rest of this codebase's CSD call (space/pipeline.py::csd) already uses.
    (s0_wm, s0_gm, s0_csf), (tr2_wm, tr1_gm, tr1_csf), _selection = three_tissue_response_dhollander16(
        scheme, data, wm_algorithm="tournier07")
    response_seconds = time.perf_counter() - t0

    mc = MultiCompartmentSphericalHarmonicsModel(
        models=[tr2_wm, tr1_gm, tr1_csf], S0_tissue_responses=[s0_wm, s0_gm, s0_csf], sh_order=sh_order)
    _patch_tissue_scheme_check(mc)

    fit_mask = mask if voxel_positions is None else voxel_positions
    t0 = time.perf_counter()
    fitted = mc.fit(scheme, data, mask=fit_mask, solver=solver, verbose=False, use_parallel_processing=False)
    fit_seconds = time.perf_counter() - t0

    sh = np.asarray(fitted.fitted_parameters["sh_coeff"], dtype=np.float32)
    fractions = np.stack([
        np.asarray(fitted.fitted_parameters["partial_volume_0"], dtype=np.float32),
        np.asarray(fitted.fitted_parameters["partial_volume_1"], dtype=np.float32),
        np.asarray(fitted.fitted_parameters["partial_volume_2"], dtype=np.float32),
    ], axis=-1)
    responses = {"S0_wm": float(s0_wm), "S0_gm": float(s0_gm), "S0_csf": float(s0_csf),
                 "response_seconds": response_seconds, "fit_seconds": fit_seconds,
                 "n_voxels_fit": int(np.sum(fit_mask))}
    return sh, fractions, responses, scheme


def _git_sha(repo_dir):
    try:
        return subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _dmipy_fit_commit():
    """The git commit of whichever ``dmipy-fit`` is actually importable (found from the package's own file path,
    not a hard-coded location: ``PYTHONPATH`` picks which checkout this is)."""
    import dmipy_fit
    return _git_sha(os.path.dirname(os.path.dirname(os.path.abspath(dmipy_fit.__file__))))


def build_manifest(*, subject, scan, source_doi, grid_shape, voxel_size_mm, affine, bvals, bvecs, te_s, delta_s,
                    delta_cap_s, dmipy_fit_commit, disco_space_commit, solver, n_regions):
    """The asset's ``manifest.json`` payload (see disco-space#8's schema in the issue body)."""
    return {
        "subject": subject,
        "source": {"dataset": "OpenNeuro ds003416", "doi": source_doi,
                    "paper_doi": "10.1002/mrm.28926", "license": "CC0", "scan": scan},
        "grid": {"shape": list(int(s) for s in grid_shape), "voxel_size_mm": list(float(v) for v in voxel_size_mm),
                 "affine": np.asarray(affine, dtype=float).tolist()},
        "protocol": {"bvals_s_mm2": np.asarray(bvals, dtype=float).tolist(),
                     "bvecs": np.asarray(bvecs, dtype=float).tolist(), "bvec_frame": "image",
                     "TE_s": te_s, "delta_s": delta_s, "Delta_s": delta_cap_s},
        "fod": {"basis": "dmipy-fit tournier07 (dipy real_sh_tournier real-SH ordering)", "lmax": SH_ORDER,
                "frame": "image"},
        "fractions": ["wm", "gm", "csf"],
        "reconstruction": {"tool": "dmipy-fit", "commit": dmipy_fit_commit,
                            "responses": "three_tissue_response_dhollander16", "solver": solver, "sh_order": SH_ORDER},
        "parcellation": {"tool": "SynthSeg --parc --robust (BBillot/SynthSeg) + dipy rigid registration "
                                   "(mutual information) + majority vote onto the diffusion grid",
                          "atlas": "Desikan-Killiany + aseg (fs_default 84)", "n_regions": n_regions},
        "built_by": {"disco_space_commit": disco_space_commit, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }


def save_checks_sample(out_dir, data, bvals, sh, fractions, mask, n_sample=2000, seed=0):
    """``checks/msmt_sample.npz``: a fixed-seed random sample of the input signal + fitted SH coefficients +
    fractions, so dmipy-fit#39's fast solver can be checked against this (slow, cvxpy) reference."""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    n_sample = min(n_sample, len(idx))
    sel = idx[rng.choice(len(idx), size=n_sample, replace=False)]
    pos = tuple(sel.T)
    os.makedirs(os.path.join(out_dir, "checks"), exist_ok=True)
    np.savez(os.path.join(out_dir, "checks", "msmt_sample.npz"),
              voxel_index=sel.astype(np.int32), signal=data[pos].astype(np.float32),
              bvals_s_mm2=np.asarray(bvals, dtype=np.float32),
              sh_coeff=sh[pos].astype(np.float32), fractions=fractions[pos].astype(np.float32), seed=seed)


def fraction_sanity(fractions, mask):
    """The fraction-sum sanity numbers to report: the median and 99th percentile of ``wm + gm + csf`` in-mask (it
    is not constrained to 1 by the unity constraint being off for ``S0_tissue_responses``-scaled fits -- see
    ``MultiCompartmentSphericalHarmonicsModel.fit``'s ``unity_constraint='kernel_dependent'`` default), and the
    count of voxels with WM fraction > 0.5."""
    s = fractions[mask].sum(axis=-1).astype(np.float64)
    return {
        "sum_median": float(np.median(s)),
        "sum_p99": float(np.percentile(s, 99)),
        "n_wm_gt_half": int(np.sum(fractions[..., 0][mask] > 0.5)),
        "n_mask": int(mask.sum()),
    }


def fod_peak_counts(sh, wm_mask, sh_order=SH_ORDER, relative_peak_threshold=0.5, min_separation_angle=25):
    """The number of FOD peaks per WM voxel (dipy's standard ``peak_directions`` on a fixed discrete sphere, the
    tournier07 SH basis evaluated at its vertices): ``{n_peaks: count}``, the single-vs-crossing distribution."""
    from collections import Counter
    from dipy.data import get_sphere
    from dipy.direction.peaks import peak_directions
    from dmipy_tract import sh_matrix

    sphere = get_sphere(name="repulsion724")
    basis = sh_matrix(sh_order, sphere.vertices)
    coef = np.nan_to_num(np.asarray(sh, np.float64))[wm_mask]
    amplitudes = coef @ basis.T
    counts = Counter()
    for odf in amplitudes:
        odf = np.clip(odf, 0, None)
        if not np.any(odf > 0):
            counts[0] += 1
            continue
        _, vals, _ = peak_directions(odf, sphere, relative_peak_threshold=relative_peak_threshold,
                                      min_separation_angle=min_separation_angle)
        counts[len(vals)] += 1
    return dict(counts)


def save_mean_b0_fod_png(out_path, mean_b0_volume, sh, wm_mask, z=None, sh_order=SH_ORDER, step=2):
    """A PNG of one axial slice of the mean b = 0 image with the WM FOD's principal direction overlaid as a short
    line segment per voxel (every ``step``-th voxel, to keep the plot legible); Matplotlib's ``Agg`` backend, no
    display needed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dmipy_tract import hemisphere, sh_matrix

    z = mean_b0_volume.shape[2] // 2 if z is None else z
    dirs = hemisphere(362)
    basis = sh_matrix(sh_order, dirs)
    coef = np.nan_to_num(np.asarray(sh, np.float64)).reshape(-1, sh.shape[-1])
    amp = coef @ basis.T
    k = np.argmax(amp, axis=1)
    peak = np.take_along_axis(amp, k[:, None], axis=1)[:, 0]
    principal = np.where(peak[:, None] > 0, dirs[k], 0.0).reshape(sh.shape[:-1] + (3,))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mean_b0_volume[:, :, z].T, cmap="gray", origin="lower")
    xs, ys, us, vs = [], [], [], []
    for x in range(0, mean_b0_volume.shape[0], step):
        for y in range(0, mean_b0_volume.shape[1], step):
            if not wm_mask[x, y, z]:
                continue
            d = principal[x, y, z]
            if np.linalg.norm(d) < 1e-6:
                continue
            xs.append(x); ys.append(y); us.append(d[0]); vs.append(d[1])
    ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1.2, headaxislength=0, headlength=0,
              width=0.003, color="red")
    ax.set_title(f"mean b=0, axial z={z}, WM FOD principal direction")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_checks(out_dir):
    """Loads a finished asset directory's arrays and writes ``checks/sanity_summary.json`` and
    ``checks/mean_b0_fod_slice.png`` (the peak-count / fraction-sum / WM-volume numbers this issue asks to report,
    decoupled from the -- long -- fit itself, so a completed fit's checks can be (re)computed on their own)."""
    fractions = np.load(os.path.join(out_dir, "fractions.npy")).astype(np.float32)
    mask = np.load(os.path.join(out_dir, "mask.npy"))
    sh = np.load(os.path.join(out_dir, "fod_wm.npy"))
    mean_b0_volume = np.load(os.path.join(out_dir, "mean_b0.npy"))
    wm_mask = mask & (fractions[..., 0] > 0.5)

    summary = {"fractions": fraction_sanity(fractions, mask), "fod_peak_counts_in_wm": fod_peak_counts(sh, wm_mask)}
    os.makedirs(os.path.join(out_dir, "checks"), exist_ok=True)
    with open(os.path.join(out_dir, "checks", "sanity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    save_mean_b0_fod_png(os.path.join(out_dir, "checks", "mean_b0_fod_slice.png"), mean_b0_volume, sh, wm_mask)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dwi-dir", required=False, default=None)
    ap.add_argument("--parcellation-dir", required=False, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subject", default="MASiVar sub-cIs1 ses-s1Ax1 run-1xx")
    ap.add_argument("--source-doi", default="10.18112/openneuro.ds003416.v2.0.2")
    ap.add_argument("--solver", default="csd_cvxpy")
    ap.add_argument("--n-test-voxels", type=int, default=None,
                     help="fit only this many random masked voxels (a dry run; nothing written under --out-dir "
                          "except the printed timing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checks-only", action="store_true",
                     help="skip everything above and just (re)compute checks/sanity_summary.json + the FOD PNG "
                          "from an --out-dir a previous run already wrote")
    a = ap.parse_args()

    if a.checks_only:
        run_checks(a.out_dir)
        return
    if not a.dwi_dir or not a.parcellation_dir:
        ap.error("--dwi-dir and --parcellation-dir are required unless --checks-only")

    import nibabel as nib

    data, bvals, bvecs, affine, shells = merge_shells(a.dwi_dir)
    b0 = mean_b0(data, bvals)
    mask = brain_mask(b0)
    voxel_size_mm = np.linalg.norm(affine[:3, :3], axis=0)
    print(f"grid {data.shape[:3]}, voxel {voxel_size_mm} mm, {data.shape[-1]} measurements, shells {shells}, "
          f"mask voxels {mask.sum()}")

    if a.n_test_voxels:
        rng = np.random.default_rng(a.seed)
        idx = np.argwhere(mask)
        sel = idx[rng.choice(len(idx), size=min(a.n_test_voxels, len(idx)), replace=False)]
        voxel_positions = np.zeros(mask.shape, dtype=bool)
        voxel_positions[tuple(sel.T)] = True
        sh, fractions, responses, _scheme = fit_msmt_csd(
            data, bvals, bvecs, mask, solver=a.solver, voxel_positions=voxel_positions)
        per_voxel = responses["fit_seconds"] / responses["n_voxels_fit"]
        print(f"responses {responses['response_seconds']:.1f} s; fit {responses['n_voxels_fit']} voxels in "
              f"{responses['fit_seconds']:.1f} s ({per_voxel * 1e3:.1f} ms/voxel); whole brain "
              f"({int(mask.sum())} voxels) extrapolates to {per_voxel * int(mask.sum()) / 60:.1f} min serial")
        return

    labels = np.load(os.path.join(a.parcellation_dir, "labels.npy"))
    stop_mask = np.load(os.path.join(a.parcellation_dir, "stop_mask.npy"))
    with open(os.path.join(a.parcellation_dir, "regions.json")) as f:
        regions = json.load(f)
    if labels.shape != data.shape[:3]:
        raise ValueError(f"parcellation grid {labels.shape} does not match the DWI grid {data.shape[:3]}")

    sh, fractions, responses, _scheme = fit_msmt_csd(data, bvals, bvecs, mask, solver=a.solver)
    print(f"responses {responses['response_seconds']:.1f} s; whole-brain fit "
          f"{responses['n_voxels_fit']} voxels in {responses['fit_seconds']:.1f} s")

    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "fod_wm.npy"), sh)
    np.save(os.path.join(a.out_dir, "fractions.npy"), np.clip(fractions, 0, 1).astype(np.float16))
    np.save(os.path.join(a.out_dir, "mask.npy"), mask)
    np.save(os.path.join(a.out_dir, "mean_b0.npy"), b0)
    np.save(os.path.join(a.out_dir, "labels.npy"), labels)
    np.save(os.path.join(a.out_dir, "stop_mask.npy"), stop_mask)
    with open(os.path.join(a.out_dir, "regions.json"), "w") as f:
        json.dump(regions, f, indent=2)

    manifest = build_manifest(
        subject=a.subject, scan="ses-s1Ax1 run-1xx (b1000/1500/2000/2500/3000, n=96 each, PreQual-preprocessed)",
        source_doi=a.source_doi, grid_shape=data.shape[:3], voxel_size_mm=voxel_size_mm, affine=affine,
        bvals=bvals, bvecs=bvecs, te_s=None, delta_s=None, delta_cap_s=None,
        dmipy_fit_commit=_dmipy_fit_commit(),
        disco_space_commit=_git_sha(os.path.join(os.path.dirname(__file__), "..")),
        solver=a.solver, n_regions=len(regions["regions"]))
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    save_checks_sample(a.out_dir, data, bvals, sh, fractions, mask, seed=a.seed)
    run_checks(a.out_dir)
    print(f"asset written to {a.out_dir}")


if __name__ == "__main__":
    main()
