"""``build_brain_asset.py``'s pieces, on tiny synthetic data: merging shells, the mean b=0, the manifest payload and
the checks sample are pure numpy/nibabel and run in every CI; the dmipy-fit MSMT-CSD fit itself needs cvxpy and is
skipped when it is not installed (the offline build's own venv; not part of the Space's requirements.txt)."""
import json
import os

import nibabel as nib
import numpy as np
import pytest

from tools import build_brain_asset as BA


def _write_shell(tmp_path, stem, shape, bval, n_dirs, affine, seed):
    rng = np.random.default_rng(seed)
    n = n_dirs + 1
    data = rng.random(shape + (n,)).astype(np.float32) + 1.0
    nib.Nifti1Image(data, affine).to_filename(os.path.join(tmp_path, stem + ".nii.gz"))
    bvals = np.r_[np.full(n_dirs, float(bval)), 0.0]
    dirs = rng.normal(size=(n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvecs = np.vstack([dirs, [[0.0, 0.0, 0.0]]])
    np.savetxt(os.path.join(tmp_path, stem + ".bval"), bvals[None, :], fmt="%g")
    np.savetxt(os.path.join(tmp_path, stem + ".bvec"), bvecs.T, fmt="%g")
    return data, bvals, bvecs


def test_merge_shells_concatenates_and_orders(tmp_path):
    affine = np.diag([2.5, 2.5, 2.5, 1.0])
    shape = (4, 4, 3)
    d1, bv1, vec1 = _write_shell(tmp_path, "a_b1000n4_dwi", shape, 1000, 4, affine, seed=0)
    d2, bv2, vec2 = _write_shell(tmp_path, "b_b2000n6_dwi", shape, 2000, 6, affine, seed=1)

    data, bvals, bvecs, aff, shells = BA.merge_shells(str(tmp_path))

    assert data.shape == shape + (5 + 7,)
    assert bvals.shape == (12,)
    assert bvecs.shape == (12, 3)
    np.testing.assert_allclose(aff, affine)
    assert shells == [(1000.0, 4), (2000.0, 6)]
    # every file's own b0 (last column) survives the concatenation
    assert np.sum(bvals < BA.B0_THRESHOLD_S_MM2) == 2


def test_merge_shells_rejects_a_mismatched_grid(tmp_path):
    affine = np.diag([2.5, 2.5, 2.5, 1.0])
    _write_shell(tmp_path, "a_b1000n2_dwi", (4, 4, 3), 1000, 2, affine, seed=0)
    _write_shell(tmp_path, "b_b2000n2_dwi", (5, 4, 3), 2000, 2, affine, seed=1)
    with pytest.raises(ValueError):
        BA.merge_shells(str(tmp_path))


def test_mean_b0_averages_only_the_b0_volumes():
    bvals = np.array([0.0, 1000.0, 1000.0, 10.0])
    data = np.zeros((2, 2, 2, 4), dtype=np.float32)
    data[..., 0] = 2.0
    data[..., 3] = 4.0
    data[..., 1] = 999.0
    data[..., 2] = 999.0
    out = BA.mean_b0(data, bvals)
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out, 3.0)


def test_mean_b0_raises_without_a_b0():
    with pytest.raises(ValueError):
        BA.mean_b0(np.zeros((2, 2, 2, 3)), np.array([1000.0, 1000.0, 1000.0]))


def test_build_manifest_schema():
    m = BA.build_manifest(
        subject="s", scan="sc", source_doi="10.x/y", grid_shape=(2, 3, 4), voxel_size_mm=(2.5, 2.5, 2.5),
        affine=np.eye(4), bvals=np.array([0.0, 1000.0]), bvecs=np.zeros((2, 3)), te_s=None, delta_s=None,
        delta_cap_s=None, dmipy_fit_commit="abc123", disco_space_commit="def456", solver="csd_cvxpy", n_regions=84)
    json.dumps(m)  # every field must be JSON-serialisable
    assert m["source"]["dataset"] == "OpenNeuro ds003416"
    assert m["source"]["paper_doi"] == "10.1002/mrm.28926"
    assert m["source"]["license"] == "CC0"
    assert m["grid"]["shape"] == [2, 3, 4]
    assert m["fod"]["lmax"] == BA.SH_ORDER
    assert m["fractions"] == ["wm", "gm", "csf"]
    assert m["reconstruction"]["tool"] == "dmipy-fit"
    assert m["reconstruction"]["commit"] == "abc123"
    assert m["parcellation"]["n_regions"] == 84


def test_save_checks_sample_is_reproducible(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.random((5, 5, 5, 7)).astype(np.float32)
    sh = rng.random((5, 5, 5, 45)).astype(np.float32)
    fractions = rng.random((5, 5, 5, 3)).astype(np.float32)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    bvals = np.linspace(0, 3000, 7)

    BA.save_checks_sample(str(tmp_path), data, bvals, sh, fractions, mask, n_sample=10, seed=0)
    out = np.load(os.path.join(tmp_path, "checks", "msmt_sample.npz"))
    assert out["voxel_index"].shape == (10, 3)
    assert out["signal"].shape == (10, 7)
    assert out["sh_coeff"].shape == (10, 45)
    assert out["fractions"].shape == (10, 3)
    for idx, sig in zip(out["voxel_index"], out["signal"]):
        assert mask[tuple(idx)]
        np.testing.assert_allclose(sig, data[tuple(idx)])

    BA.save_checks_sample(str(tmp_path), data, bvals, sh, fractions, mask, n_sample=10, seed=0)
    out2 = np.load(os.path.join(tmp_path, "checks", "msmt_sample.npz"))
    np.testing.assert_array_equal(out["voxel_index"], out2["voxel_index"])


def test_fraction_sanity():
    fractions = np.zeros((2, 2, 1, 3), dtype=np.float32)
    fractions[0, 0, 0] = [0.7, 0.2, 0.1]     # sum 1.0, WM > 0.5
    fractions[0, 1, 0] = [0.3, 0.3, 0.3]     # sum 0.9, WM not > 0.5
    fractions[1, 0, 0] = [0.9, 0.05, 0.05]   # sum 1.0, WM > 0.5
    mask = np.zeros((2, 2, 1), dtype=bool)
    mask[0, 0, 0] = mask[0, 1, 0] = mask[1, 0, 0] = True   # (1, 1, 0) left out of the mask on purpose

    out = BA.fraction_sanity(fractions, mask)
    assert out["n_mask"] == 3
    assert out["n_wm_gt_half"] == 2
    assert out["sum_median"] == pytest.approx(1.0)


def _pure_direction_sh(sh_order, direction):
    """The real-SH coefficients of a single delta-like lobe along ``direction`` (dipy's own ``sf_to_sh`` on a
    one-hot signal at a fine sphere, so the test does not depend on dmipy-fit's basis machinery)."""
    from dipy.data import get_sphere
    from dipy.reconst.shm import sf_to_sh
    sphere = get_sphere(name="repulsion724")
    cos_angle = sphere.vertices @ np.asarray(direction) / np.linalg.norm(direction)
    sf = np.clip(cos_angle, 0, None) ** 16   # a sharp lobe peaked at `direction`
    return sf_to_sh(sf, sphere, sh_order_max=sh_order, basis_type="tournier07")


def test_fod_peak_counts_single_vs_crossing():
    sh_order = 8
    single = _pure_direction_sh(sh_order, (0, 0, 1))
    crossing = _pure_direction_sh(sh_order, (1, 0, 0)) + _pure_direction_sh(sh_order, (0, 1, 0))
    sh = np.stack([single, crossing]).reshape(1, 2, 1, -1).astype(np.float32)
    wm_mask = np.ones((1, 2, 1), dtype=bool)

    counts = BA.fod_peak_counts(sh, wm_mask, sh_order=sh_order)
    assert sum(counts.values()) == 2
    assert counts.get(1, 0) >= 1   # the single lobe
    assert counts.get(2, 0) >= 1   # the two well-separated lobes


def test_save_mean_b0_fod_png_writes_a_file(tmp_path):
    shape = (4, 4, 2)
    mean_b0_volume = np.random.default_rng(0).random(shape).astype(np.float32)
    sh = np.zeros(shape + (45,), dtype=np.float32)
    sh[..., 0] = 1.0   # an isotropic (but non-zero) FOD everywhere: a valid, if uninteresting, principal direction
    wm_mask = np.ones(shape, dtype=bool)
    out_path = os.path.join(tmp_path, "slice.png")

    BA.save_mean_b0_fod_png(out_path, mean_b0_volume, sh, wm_mask, z=0)
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0


def _single_tensor_attenuation(bvals_s_mm2, bvecs, eigenvalues_mm2_s, axis):
    """A closed-form single-tensor signal attenuation ``exp(-b g^T D g)``, no dmipy-sim/jax needed: ``D`` has its
    principal eigenvector along ``axis`` (unit vector) with ``eigenvalues_mm2_s = (lambda_par, lambda_perp)``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    lambda_par, lambda_perp = eigenvalues_mm2_s
    d_par_perp = lambda_par - lambda_perp
    cos2 = (bvecs @ axis) ** 2
    return np.exp(-bvals_s_mm2 * (lambda_perp + d_par_perp * cos2))


def test_fit_msmt_csd_end_to_end_on_a_tiny_synthetic_volume():
    """A smoke test of the dmipy-fit call itself (skipped if cvxpy is not installed): a few voxels of a synthetic
    single-tensor-plus-isotropic signal (plain numpy, no dmipy-sim), checking only shapes and that the fractions
    come out non-negative."""
    pytest.importorskip("cvxpy")

    rng = np.random.default_rng(0)
    n_dirs = 30
    dirs = rng.normal(size=(n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvals_shell = np.repeat([1000.0, 2000.0], n_dirs)
    bvecs_shell = np.tile(dirs, (2, 1))
    bvals = np.r_[0.0, 0.0, bvals_shell]
    bvecs = np.vstack([[0, 0, 0], [0, 0, 0], bvecs_shell])

    # dhollander16 classifies every voxel by FA and a signal-decay metric, then takes small percentiles (2 %, 10 %)
    # of the GM-like / CSF-like populations as its candidates: a few discrete classes risk one of those percentiles
    # rounding to an empty set (a NaN response, poisoning the CSD kernel), so the volume instead has a continuum of
    # randomly oriented, randomly scaled tensors spanning WM-like (high FA) through GM-like to CSF-like (isotropic,
    # high diffusivity) -- much closer to what real brain data actually looks like to this heuristic.
    shape = (12, 12, 6)
    n_voxels = int(np.prod(shape))
    fa_like = rng.uniform(0.0, 1.0, size=n_voxels)               # 0 = isotropic, 1 = maximally anisotropic
    mean_d = np.where(fa_like < 0.5, rng.uniform(0.7e-3, 1.0e-3, n_voxels),   # GM/WM-ish mean diffusivity
                       rng.uniform(1.0e-3, 1.4e-3, n_voxels))
    is_csf = rng.uniform(0, 1, n_voxels) < 0.15
    mean_d = np.where(is_csf, rng.uniform(2.8e-3, 3.2e-3, n_voxels), mean_d)
    fa_like = np.where(is_csf, 0.0, fa_like)
    lambda_perp = mean_d * (1.0 - 0.6 * fa_like)
    lambda_par = mean_d * (1.0 + 1.2 * fa_like)
    axes = rng.normal(size=(n_voxels, 3))

    mask = np.ones(shape, dtype=bool)
    data = np.empty((n_voxels, len(bvals)), dtype=np.float64)
    for i in range(n_voxels):
        data[i] = _single_tensor_attenuation(bvals, bvecs, (lambda_par[i], lambda_perp[i]), axis=axes[i])
    data *= 1.0 + 0.01 * rng.standard_normal(data.shape)
    data = data.reshape(shape + (len(bvals),))

    voxel_positions = np.zeros(shape, dtype=bool)
    voxel_positions[0, 0, 0] = voxel_positions[1, 0, 0] = voxel_positions[2, 0, 0] = voxel_positions[3, 0, 0] = True

    sh, fractions, responses, out_scheme = BA.fit_msmt_csd(
        data, bvals, bvecs, mask, sh_order=4, voxel_positions=voxel_positions)
    assert sh.shape == shape + (15,)          # lmax = 4 -> 15 real-SH coefficients
    assert fractions.shape == shape + (3,)
    assert responses["n_voxels_fit"] == voxel_positions.sum() == 4
    assert np.all(fractions[voxel_positions] >= -1e-4)      # non-negative to the QP solver's feasibility tolerance
    assert np.all(fractions[~voxel_positions] == 0)  # only the requested voxels were fit
