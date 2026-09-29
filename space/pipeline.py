"""The DiSCo Space's pipeline: an acquisition on the replay layout -> the DWI of the 40^3 grid -> noise -> CSD ->
probabilistic tracking from the sixteen regions -> the 16 x 16 connectome against the dataset's ground truth. Plain
functions with their timings; nothing here draws or reads a widget.

The data path is dmipy-sim's shape-moment layout (``dmipy_sim.replay.shape_moments``): the DiSCo replay pack
contracted once per PGSE timing class ("shape"), so a protocol on those classes at any b-values and directions is an
elementwise kernel on device-resident rows. The CSD is dmipy-fit's ``csd_tournier07_jax`` at order 8 with the
response from the volume's single-fibre voxels; the tracker is dmipy-tract's ``track``; the score is the Pearson
correlation of the symmetrised streamline counts with the strand-count and cross-sectional-area matrices over the
120 region pairs, as the replay paper's ``disco_tract.py`` scores.
"""
from __future__ import annotations

import os
import time
try:
    import tomllib                                   # 3.11+
except ModuleNotFoundError:                          # the ZeroGPU image is Python 3.10
    import tomli as tomllib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(HERE, "config.toml")
DATA_DIR = os.path.join(ROOT, "data")
SH_ORDER = 8
N_REGIONS = 16


def config(path=CONFIG_PATH):
    with open(path, "rb") as f:
        return tomllib.load(f)


BACKENDS = ("jax", "torch")


def resident(cfg):
    """Whether the layout's device tensors stay across calls: ``DISCO_RESIDENT`` (``0``/``false`` for a pool that
    drops the device between calls) else the config's ``[compute] resident`` (true)."""
    env = os.environ.get("DISCO_RESIDENT")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no")
    return bool(cfg.get("compute", {}).get("resident", True))


def backend(cfg):
    """The compute backend: ``DISCO_BACKEND`` in the environment, else the config's ``[compute] backend`` (``jax``).
    ``torch`` is for a host that runs PyTorch only (disco-space#2): the layout image, the CSD solver and the tracker
    all switch together."""
    b = os.environ.get("DISCO_BACKEND") or cfg.get("compute", {}).get("backend", "jax")
    if b not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {b!r}")
    return b


# ---- the acquisition -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shell:
    """One shell: the timing class it plays, its b-value (s/mm^2) and its direction count."""
    shape: str
    b: float
    n_dirs: int


@dataclass(frozen=True)
class Protocol:
    """The measurements of one acquisition: ``n_b0`` b = 0 rows on the first shell's timing, then every shell's
    directions (a Fibonacci hemisphere of ``n_dirs`` unless ``directions`` gives each row's own; ``bvals`` likewise
    gives each row's own b-value in place of the shells')."""
    shells: tuple
    n_b0: int = 1
    directions: Optional[np.ndarray] = None
    bvals: Optional[np.ndarray] = None
    name: str = "custom"

    def __post_init__(self):
        if not self.shells:
            raise ValueError("a protocol has at least one shell")
        if self.n_b0 < 1:
            raise ValueError("a protocol has at least one b = 0 measurement (the DWI is normalised by it)")
        for s in self.shells:
            if s.b <= 0 or s.n_dirs < 1:
                raise ValueError(f"a shell has b > 0 and at least one direction, got {s}")
        if self.directions is not None and self.directions.shape != (self.n_meas, 3):
            raise ValueError(f"directions must be ({self.n_meas}, 3), got {self.directions.shape}")
        if self.bvals is not None and self.bvals.shape != (self.n_meas,):
            raise ValueError(f"bvals must be ({self.n_meas},), got {self.bvals.shape}")

    @property
    def n_meas(self):
        return self.n_b0 + sum(s.n_dirs for s in self.shells)


@dataclass(frozen=True)
class Measurements:
    """Per row: b (s/mm^2), unit direction, timing-class name, delta / Delta / TE (s)."""
    bvals: np.ndarray
    dirs: np.ndarray
    shape: np.ndarray
    delta: np.ndarray
    Delta: np.ndarray
    TE: np.ndarray

    @property
    def b0(self):
        return self.bvals < 50


def measurements(protocol, shapes):
    """The rows of ``protocol`` with the timings of ``shapes`` (the config's ``[shapes]`` table)."""
    from dmipy_tract import hemisphere
    for s in protocol.shells:
        if s.shape not in shapes:
            raise KeyError(f"no timing class {s.shape!r}; the layout holds {sorted(shapes)}")
    names = [protocol.shells[0].shape] * protocol.n_b0
    b = [0.0] * protocol.n_b0
    dirs = [np.tile([0.0, 0.0, 1.0], (protocol.n_b0, 1))]
    for s in protocol.shells:
        names += [s.shape] * s.n_dirs; b += [s.b] * s.n_dirs
        dirs.append(hemisphere(s.n_dirs))
    dirs = np.concatenate(dirs) if protocol.directions is None else np.asarray(protocol.directions, np.float64)
    b = np.asarray(b, np.float64) if protocol.bvals is None else np.asarray(protocol.bvals, np.float64)
    delta = np.array([shapes[n]["delta"] for n in names]); Delta = np.array([shapes[n]["Delta"] for n in names])
    TE = np.array([shapes[n]["TE"] for n in names])
    return Measurements(b, dirs, np.asarray(names), delta, Delta, TE)


def disco_protocol(cfg):
    """DiSCo's own 364-measurement protocol (its gradient table, its per-shell timings) as a Protocol with every
    row's own direction and b-value: the acquisition the reference volumes were replayed with. Returns it with the
    rows' b-values in the protocol's order."""
    bv = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_gradients.bvals")).ravel()
    dirs = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_gradients_dipy.bvecs"))
    if dirs.shape[0] == 3:
        dirs = dirs.T
    dirs = np.where(bv[:, None] > 0, dirs, np.array([0.0, 0.0, 1.0]))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)           # the text table's rounding leaves 4e-6
    shells = cfg["disco"]["shells"]                                     # [{b, shape}] in the table's row order
    centres = np.array([s["b"] for s in shells])
    which = np.argmin(np.abs(bv[:, None] - centres[None, :]), axis=1)
    b0 = bv < 50                                    # rows: the b = 0 first, then the shells in the table's order
    idx = [np.flatnonzero(b0)]
    sh = []
    for k, s in enumerate(shells):
        rows = np.flatnonzero((which == k) & ~b0)
        if len(rows):
            idx.append(rows); sh.append(Shell(s["shape"], float(np.round(bv[rows].mean())), int(len(rows))))
    idx = np.concatenate(idx)
    return Protocol(tuple(sh), n_b0=int(b0.sum()), directions=dirs[idx], bvals=bv[idx], name="DiSCo 364"), bv[idx]


def protocol_from_table(bvals_path, bvecs_path, shape, shapes, *, name="uploaded table"):
    """An uploaded gradient table (FSL bvals in s/mm^2 and bvecs, either orientation) as a Protocol whose rows all play
    the timing class ``shape``: every row's own b-value and direction, the b = 0 rows first. Refuses a malformed
    table, a class the layout does not hold, and a table without a b = 0 row."""
    if shape not in shapes:
        raise KeyError(f"no timing class {shape!r}; the layout holds {sorted(shapes)}")
    bv = np.loadtxt(bvals_path, ndmin=1).ravel().astype(np.float64)
    dirs = np.loadtxt(bvecs_path, ndmin=2).astype(np.float64)
    if dirs.shape[0] == 3 and dirs.shape[1] != 3:
        dirs = dirs.T
    if dirs.ndim != 2 or dirs.shape[1] != 3 or dirs.shape[0] != len(bv):
        raise ValueError(f"bvecs must be (N, 3) or (3, N) with N = {len(bv)} b-values, got {dirs.shape}")
    if np.any(bv < 0) or not np.all(np.isfinite(bv)) or not np.all(np.isfinite(dirs)):
        raise ValueError("the table has a negative or non-finite entry")
    b0 = bv < 50
    if not b0.any():
        raise ValueError("the table needs a b = 0 row (the DWI is normalised by it)")
    norm = np.linalg.norm(dirs, axis=1)
    if np.any(norm[~b0] == 0):
        raise ValueError("a b > 0 row has a zero direction")
    dirs = np.where(b0[:, None], np.array([0.0, 0.0, 1.0]), dirs / np.where(norm > 0, norm, 1.0)[:, None])
    order = np.r_[np.flatnonzero(b0), np.flatnonzero(~b0)]
    rows = bv[order]; d = dirs[order]
    shells_b, counts = np.unique(np.round(rows[~b0[order]]), return_counts=True)
    shells = tuple(Shell(shape, float(b), int(n)) for b, n in zip(shells_b, counts))
    return Protocol(shells, n_b0=int(b0.sum()), directions=d, bvals=rows, name=name)


def write_volumes(res, out_dir, *, prefix="disco"):
    """The result's DWI as NIfTI (float32, identity affine, NaN as 0) with its bvals (s/mm^2) and bvecs (3 x N), and
    the FOD SH field as NIfTI (tournier07 basis, order 8); returns the four paths."""
    import nibabel as nib
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    dwi = os.path.join(out_dir, f"{prefix}_dwi.nii.gz")
    nib.save(nib.Nifti1Image(np.nan_to_num(res.dwi).astype(np.float32), np.eye(4)), dwi); paths["dwi"] = dwi
    bvals = os.path.join(out_dir, f"{prefix}.bvals"); np.savetxt(bvals, res.meas.bvals[None, :], fmt="%.1f"); paths["bvals"] = bvals
    bvecs = os.path.join(out_dir, f"{prefix}.bvecs"); np.savetxt(bvecs, res.meas.dirs.T, fmt="%.8f"); paths["bvecs"] = bvecs
    fod = os.path.join(out_dir, f"{prefix}_fod_sh.nii.gz")
    nib.save(nib.Nifti1Image(np.nan_to_num(res.sh).astype(np.float32), np.eye(4)), fod); paths["fod"] = fod
    return paths


# ---- the data ---------------------------------------------------------------------------------------------------

class Layout:
    """What the pipeline reads: the shape-moment layout (device-resident rows), the DiSCo mask, regions and the
    two ground-truth matrices."""

    def __init__(self, cfg, *, local=None):
        import nibabel as nib
        from dmipy_sim.replay.shape_moments import ShapeMoments
        d = cfg["data"]
        uri = local or f"hf://{d['repo']}/{d['moments']}"
        self.moments = ShapeMoments.open(uri, revision=d.get("revision") or None) if uri.startswith("hf://") else ShapeMoments(uri)
        self.shapes = cfg["shapes"]
        self.backend = backend(cfg)
        self.resident = resident(cfg)
        want = d.get("source_manifest_sha256")
        got = self.moments.manifest["source"]["manifest_sha256"]
        if want and got != want:
            raise ValueError(f"the layout at {uri} was contracted from columnar manifest {got[:12]}, the config pins {want[:12]}")
        missing = [s for s in self.shapes if s not in self.moments.shapes]
        if missing:
            raise ValueError(f"the layout at {uri} lacks the timing classes {missing}; it holds {self.moments.shapes}")
        self.mask = np.asarray(nib.load(os.path.join(DATA_DIR, "DiSCo_mask.nii.gz")).dataobj) > 0
        self.rois = np.asarray(nib.load(os.path.join(DATA_DIR, "DiSCo_ROIs.nii.gz")).dataobj).astype(np.int32)
        self.gt_count = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_Connectivity_Matrix_Strands_Count.txt"))
        self.gt_area = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_Connectivity_Matrix_Cross-Sectional_Area.txt"))
        if tuple(self.moments.grid.shape) != self.mask.shape:
            raise ValueError(f"the layout's grid {tuple(self.moments.grid.shape)} is not the mask's {self.mask.shape}")

    def warm(self):
        """Every timing class on the device (the first call to each compiles and transfers); when the tiles are not
        kept resident (a pool that drops the device between calls), the padded host arrays are loaded into this
        process instead, so a forked worker inherits them and pays the transfer alone."""
        if not self.resident:
            self.moments.preload(list(self.shapes))
            return
        for s in self.shapes:
            self.moments.image(s, [0.0], [[0.0, 0.0, 1.0]], backend=self.backend)


# ---- the stages -------------------------------------------------------------------------------------------------

def replay(layout, meas):
    """``(dwi, floor, seconds)``: the S0-normalised signal of every voxel per measurement (NaN outside the pack's
    rows), the largest split-half floor over the timing classes, and the time."""
    t0 = time.perf_counter()
    S = np.full(layout.mask.shape + (len(meas.bvals),), np.nan)
    floor = np.zeros(layout.mask.shape)
    for name in np.unique(meas.shape):
        rows = meas.shape == name
        S[..., rows], f = layout.moments.image(name, meas.bvals[rows] * 1e6, meas.dirs[rows], backend=layout.backend,
                                                resident=layout.resident)
        floor = np.fmax(floor, f)
    S0 = np.nanmean(S[..., meas.b0], axis=-1, keepdims=True)
    return S / S0, floor, time.perf_counter() - t0


def add_noise(dwi, snr, seed=0, backend="jax"):
    """Rician noise at ``snr`` (the b = 0 SNR; the DWI is S0-normalised, so sigma = 1 / snr); ``None`` leaves the
    signal noiseless. NaN voxels stay NaN. The draw comes from JAX's generator on the jax backend and from numpy's
    on the torch backend (a forked GPU worker must not touch JAX); the two streams differ, the distribution is the
    same."""
    if snr is None:
        return dwi
    if snr <= 0:
        raise ValueError("SNR is positive, or None for no noise")
    from dmipy_sim.acquisition.noise import add_rician_noise
    valid = np.isfinite(dwi)
    out = np.array(dwi)
    out[valid] = np.asarray(add_rician_noise(dwi[valid], 1.0 / snr, seed=seed, rng="numpy" if backend == "torch" else "jax"))
    return out


def scheme(meas):
    from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
    return acquisition_scheme_from_bvalues(meas.bvals * 1e6, meas.dirs, delta=meas.delta, Delta=meas.Delta, TE=meas.TE, b0_threshold=10e6)


def csd(dwi, meas, mask, backend="jax"):
    """``(sh, seconds)``: the FOD field ``(X, Y, Z, 45)`` in the tournier07 basis from the single-fibre response of
    the volume, dmipy-fit's batched Tournier 2007 solver on ``backend``, fitted on ``mask`` (the voxels with signal)."""
    from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
    from dmipy_fit.tissue_response.white_matter_response import white_matter_response_tournier07
    t0 = time.perf_counter()
    data = np.nan_to_num(dwi, nan=0.0)
    sch = scheme(meas)
    S0_wm, response, _ = white_matter_response_tournier07(sch, data[mask])
    mc = MultiCompartmentSphericalHarmonicsModel(models=[response], sh_order=SH_ORDER)
    fitted = mc.fit(sch, data, mask=mask, solver=f"csd_tournier07_{backend}", verbose=False)
    return np.asarray(fitted.fitted_parameters["sh_coeff"], np.float64), time.perf_counter() - t0


@dataclass(frozen=True)
class Tracking:
    """The tracker's settings: dipy's defaults for the DiSCo score."""
    density: int = 4
    step_mm: float = 0.5
    max_angle: float = 30.0
    max_steps: int = 500
    relative_threshold: float = 0.1
    key: int = 0


def track(sh, layout, settings=Tracking()):
    """``(tractogram, seeds, seconds)``: probabilistic streamlines from ``density^3`` seeds per region voxel."""
    from dmipy_tract import FODField, seeds_from_mask, track as _track
    t0 = time.perf_counter()
    field = FODField(sh.astype(np.float32), np.eye(4), layout.mask | (layout.rois > 0))
    seeds = seeds_from_mask(layout.rois > 0, np.eye(4), density=settings.density)
    tg = _track(field, seeds, rule="probabilistic", step_mm=settings.step_mm, max_angle=settings.max_angle,
                max_steps=settings.max_steps, relative_threshold=settings.relative_threshold, key=settings.key,
                backend=layout.backend)
    return tg, seeds, time.perf_counter() - t0


def peaks(sh, n_dirs=362):
    """The principal direction of every voxel's FOD, ``(X, Y, Z, 3)`` unit vectors, and its amplitude ``(X, Y, Z)``:
    the largest FOD value over the tracker's hemisphere (zero where the FOD is zero or NaN)."""
    from dmipy_tract import hemisphere, sh_matrix
    dirs = hemisphere(n_dirs)
    B = sh_matrix(SH_ORDER, dirs)                                     # (n_dirs, n_coef)
    coef = np.nan_to_num(np.asarray(sh, np.float64)).reshape(-1, sh.shape[-1])
    amp = coef @ B.T
    k = np.argmax(amp, axis=1)
    peak = np.take_along_axis(amp, k[:, None], axis=1)[:, 0]
    d = np.where(peak[:, None] > 0, dirs[k], 0.0)
    return d.reshape(sh.shape[:-1] + (3,)), np.clip(peak, 0, None).reshape(sh.shape[:-1])


def connectome(tg, layout):
    """The symmetrised 16 x 16 streamline-count matrix (no self-connections)."""
    from dmipy_tract import connectivity
    matrix, _ = connectivity(tg, layout.rois, np.eye(4))
    M = matrix[1:N_REGIONS + 1, 1:N_REGIONS + 1].astype(np.float64)
    M = M + M.T
    np.fill_diagonal(M, 0)
    return M


def score(M, layout):
    """Pearson correlations over the 120 region pairs with the strand-count and area matrices, and the pair
    bookkeeping (connected, ground-truth, false, missed)."""
    iu = np.triu_indices(N_REGIONS, 1)
    gt = layout.gt_count
    return dict(pearson_count=float(np.corrcoef(M[iu], gt[iu])[0, 1]),
                pearson_area=float(np.corrcoef(M[iu], layout.gt_area[iu])[0, 1]),
                connected_pairs=int((M[iu] > 0).sum()), gt_pairs=int((gt[iu] > 0).sum()),
                false_pairs=int(((M[iu] > 0) & (gt[iu] == 0)).sum()), missed_pairs=int(((M[iu] == 0) & (gt[iu] > 0)).sum()))


@dataclass
class Result:
    protocol: Protocol
    meas: Measurements
    dwi: np.ndarray
    floor: np.ndarray
    sh: np.ndarray
    tractogram: object
    seeds: np.ndarray
    matrix: np.ndarray
    score: dict
    seconds: dict = field(default_factory=dict)
    snr: Optional[float] = None


STAGES = ("replay", "noise", "csd", "track", "score")


def run(layout, protocol, *, snr=None, tracking=Tracking(), noise_seed=0, progress=None):
    """The whole pipeline for one protocol: every stage's output and time. ``progress(stage, k, n)`` is called as
    each of the ``STAGES`` begins (``k`` of ``n``), so a caller can say where the run is."""
    t_run = time.perf_counter()

    def at(stage):
        print(f"[pipeline] {stage} at +{time.perf_counter() - t_run:.1f} s", flush=True)      # the server log shows where a run is
        if progress:
            progress(stage, STAGES.index(stage), len(STAGES))
    meas = measurements(protocol, layout.shapes)
    at("replay"); dwi, floor, t_replay = replay(layout, meas)
    at("noise"); t0 = time.perf_counter(); noisy = add_noise(dwi, snr, seed=noise_seed, backend=layout.backend); t_noise = time.perf_counter() - t0
    signal = np.isfinite(dwi[..., 0])
    at("csd"); sh, t_csd = csd(noisy, meas, signal, backend=layout.backend)
    at("track"); tg, seeds, t_track = track(sh, layout, tracking)
    at("score"); t0 = time.perf_counter(); M = connectome(tg, layout); s = score(M, layout); t_score = time.perf_counter() - t0
    return Result(protocol, meas, noisy, floor, sh, tg, seeds, M, s, snr=snr,
                  seconds=dict(replay=t_replay, noise=t_noise, csd=t_csd, track=t_track, score=t_score,
                               total=t_replay + t_noise + t_csd + t_track + t_score))
