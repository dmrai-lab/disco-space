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
MODES = ("demo", "full")


def mode(cfg):
    """``demo``: the prebaked shape-moment layout (seconds; the stored pulse timings, the free knobs), what the hosted
    Spaces run. ``full``: the columnar replay pack itself (minutes per volume; any timing, any waveform, every tier),
    what the same image does with the pack beside it. ``DISCO_MODE`` in the environment, else ``[mode] mode``."""
    m = os.environ.get("DISCO_MODE") or cfg.get("mode", {}).get("mode", "demo")
    if m not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {m!r}")
    return m


def columns_uri(cfg):
    """Where full mode reads the columnar pack: ``DISCO_COLUMNS`` (a directory, or ``hf://owner/name/prefix`` at the
    Hub's 45 MB/s), else ``[mode] columns``."""
    return os.environ.get("DISCO_COLUMNS") or cfg.get("mode", {}).get("columns", "hf://SubstrateCommons/disco-replay/disco")


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
    """One shell: the timing class it plays, its b-value (s/mm^2) and its direction count. In full mode a shell may
    carry its own pulse timing instead of a stored class: ``delta``, ``Delta``, ``TE`` in seconds (``shape`` then
    names it only)."""
    shape: str
    b: float
    n_dirs: int
    delta: Optional[float] = None
    Delta: Optional[float] = None
    TE: Optional[float] = None

    @property
    def free_timing(self):
        return self.delta is not None


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
    timing = {}
    for s in protocol.shells:
        if s.free_timing:
            if s.Delta is None or s.TE is None or not (0 < s.delta < s.Delta < s.TE):
                raise ValueError(f"a free pulse timing needs 0 < delta < Delta < TE, got {s}")
            timing[s.shape] = dict(delta=float(s.delta), Delta=float(s.Delta), TE=float(s.TE))
        elif s.shape not in shapes:
            raise KeyError(f"no timing class {s.shape!r}; the layout holds {sorted(shapes)}")
        else:
            timing[s.shape] = shapes[s.shape]
    names = [protocol.shells[0].shape] * protocol.n_b0
    b = [0.0] * protocol.n_b0
    dirs = [np.tile([0.0, 0.0, 1.0], (protocol.n_b0, 1))]
    for s in protocol.shells:
        names += [s.shape] * s.n_dirs; b += [s.b] * s.n_dirs
        dirs.append(hemisphere(s.n_dirs))
    dirs = np.concatenate(dirs) if protocol.directions is None else np.asarray(protocol.directions, np.float64)
    b = np.asarray(b, np.float64) if protocol.bvals is None else np.asarray(protocol.bvals, np.float64)
    delta = np.array([timing[n]["delta"] for n in names]); Delta = np.array([timing[n]["Delta"] for n in names])
    TE = np.array([timing[n]["TE"] for n in names])
    return Measurements(b, dirs, np.asarray(names), delta, Delta, TE)


def protocol_from_scheme(path, *, name="uploaded scheme"):
    """A Camino ``STEJSKALTANNER`` scheme file (``gx gy gz |G| Delta delta TE`` per row, SI units) as a Protocol
    with every row's own b-value, direction and pulse timing, for full mode: one shell per distinct
    (b, delta, Delta, TE), the b = 0 rows first. Read by :func:`dmipy_sim.io.mcdc.read_scheme`, the one reader of
    that format."""
    from dmipy_sim.io.mcdc import read_scheme
    seq = read_scheme(path)
    enc = seq.encoding
    b = np.asarray(enc.bvalues, np.float64) / 1e6
    dirs = np.asarray(enc.gradient_directions, np.float64)
    delta = np.asarray(enc.delta, np.float64); Delta = np.asarray(enc.Delta, np.float64); TE = np.asarray(enc.TE, np.float64)
    b0 = b < 50
    if not b0.any():
        raise ValueError("the scheme needs a b = 0 row (the DWI is normalised by it)")
    dirs = np.where(b0[:, None], np.array([0.0, 0.0, 1.0]), dirs / np.where(np.linalg.norm(dirs, axis=1) > 0, np.linalg.norm(dirs, axis=1), 1.0)[:, None])
    keys = np.stack([np.round(b), np.round(delta * 1e6), np.round(Delta * 1e6), np.round(TE * 1e6)], 1)
    order = np.r_[np.flatnonzero(b0), np.flatnonzero(~b0)]
    shells = []
    for k in np.unique(keys[~b0], axis=0):
        rows = np.flatnonzero((keys == k).all(1) & ~b0)
        shells.append(Shell(f"d{k[1] / 1e3:g}-D{k[2] / 1e3:g}-TE{k[3] / 1e3:g}", float(k[0]), int(len(rows)),
                            delta=float(delta[rows[0]]), Delta=float(Delta[rows[0]]), TE=float(TE[rows[0]])))
    # the rows in the protocol's order: b = 0 first (the first shell's timing), then each shell's rows
    idx = [np.flatnonzero(b0)] + [np.flatnonzero((keys == k).all(1) & ~b0) for k in np.unique(keys[~b0], axis=0)]
    idx = np.concatenate(idx)
    return Protocol(tuple(shells), n_b0=int(b0.sum()), directions=dirs[idx], bvals=b[idx], name=name)


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

    def accuracy(self):
        """What the layout is an approximation of, from its manifest (iteration 3 of disco-space#4): the band count
        and the band error at the built amplitude, the tolerance, the source pack (id, its certificate's median
        floor, its manifest sha) and the layout's code commit; rows of ``(what, value)`` for the page."""
        m = self.moments.manifest; src = m["source"]
        rows = [["temporal bands kept (K)", str(m["K"])],
                ["band error at the built amplitude (worst class)", f"{m['band_error']:.2e}"],
                ["band tolerance (× the pack's floor)", f"{m['tol']:g}"],
                ["source pack", str(src.get("pack"))],
                ["source pack's certified median floor", f"{src['floor']:.4g}"],
                ["source manifest sha256", src["manifest_sha256"][:16]],
                ["layout written by", f"{m.get('code', {}).get('commit', '?')[:12]} on {m.get('created', '?')}"],
                ["pulses", "square (slew rate ∞), one TE per class; see the class table"],
                ["tiers stored", "pool, contact, field" if m.get("tiers") else "none (bare diffusion only)"]]
        return rows

    def warm(self):
        """Every timing class on the device (the first call to each compiles and transfers); when the tiles are not
        kept resident (a pool that drops the device between calls), the padded host arrays are loaded into this
        process instead, so a forked worker inherits them and pays the transfer alone."""
        if not self.resident:
            self.moments.preload(list(self.shapes))
            return
        for s in self.shapes:
            self.moments.image(s, [0.0], [[0.0, 0.0, 1.0]], backend=self.backend)


class Columns:
    """Full mode's data: the columnar replay pack (:class:`dmipy_sim.replay.columnar.ColumnarPack`) at
    :func:`columns_uri`, with the DiSCo mask, regions and ground truth beside it."""

    def __init__(self, cfg, *, uri=None):
        import nibabel as nib
        from dmipy_sim.replay.columnar import ColumnarPack
        self.uri = uri or columns_uri(cfg)
        self.pack = ColumnarPack(self.uri, workers=int(cfg.get("mode", {}).get("workers", 16)))
        self.shapes = cfg["shapes"]
        self.backend = "jax"; self.resident = True
        self.mask = np.asarray(nib.load(os.path.join(DATA_DIR, "DiSCo_mask.nii.gz")).dataobj) > 0
        self.rois = np.asarray(nib.load(os.path.join(DATA_DIR, "DiSCo_ROIs.nii.gz")).dataobj).astype(np.int32)
        self.gt_count = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_Connectivity_Matrix_Strands_Count.txt"))
        self.gt_area = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_Connectivity_Matrix_Cross-Sectional_Area.txt"))
        if tuple(self.pack.grid.shape) != self.mask.shape:
            raise ValueError(f"the pack's grid {tuple(self.pack.grid.shape)} is not the mask's {self.mask.shape}")
        self.remote = self.uri.startswith("hf://")

    def warm(self):
        return

    def sequence(self, meas):
        """The one ScannerSequence of the measurements (PGSE, square pulses, every row's own delta / Delta at the rows'
        common TE; several TEs are several sequences and refused here)."""
        import dmipy_sim as d
        TEs = np.unique(np.round(meas.TE, 9))
        if len(TEs) != 1:
            raise ValueError(f"full mode replays one echo time per run; the rows have {len(TEs)}")
        return d.pgse(meas.dirs.tolist(), meas.delta, meas.Delta, bvalues=meas.bvals * 1e6, TE=float(TEs[0]), n_t=1000, slew_rate=np.inf)

    def plan(self, meas, tissue=None, scanner=None):
        """What the replay will read and about how long it takes: the pack's plan plus a rate (the Hub's measured 45 MB/s,
        a local disk's 500 MB/s)."""
        plan = self.pack.plan(self.sequence(meas), tissue=tissue, scanner=scanner, tol=0.005)
        rate = 45e6 if self.remote else 500e6
        plan["estimated_seconds"] = plan["bytes"] / rate
        return plan


def replay_full(columns, meas, physics=None, *, progress=None):
    """``(dwi, floor, seconds)`` from the columnar pack: one pass over its rows for the measurements' sequence, the
    split-half floor per voxel; S0-normalised like :func:`replay`. The field's direction is the pack's z here
    (the columnar study has no orientation knob yet: disco-space#4)."""
    from dmipy_sim.replay.study import Study, Protocol as SProtocol, Acquisition
    t0 = time.perf_counter()
    tissue = physics.tissue() if physics else None
    scanner = physics.field_T if tissue is not None else None
    if physics and not np.allclose(physics.b0_direction, (0.0, 0.0, 1.0)):
        raise ValueError("full mode replays the field along z only")
    seq = columns.sequence(meas)
    study = Study(SProtocol([Acquisition(seq, name="run")]), tissues=[tissue], scanners=[scanner])
    S, floor, plan = columns.pack.image_study(study, tol=0.005, chunk_rows=1_000_000, progress=progress)
    S = S[0]; floor = floor[0]
    S0 = np.nanmean(S[..., meas.b0], axis=-1, keepdims=True)
    return S / S0, floor, time.perf_counter() - t0


# ---- the physics: tissue and scanner (disco-space#4 iteration 1) ----------------------------------------------------
FIELDS = (0.064, 1.5, 3.0, 7.0, 11.7)                    # the page's field presets (T)
CATALOGUE_FIELDS = (1.5, 3.0, 7.0)                       # where dmipy-sim's white-matter catalogue has cited relaxation
POOLS = ("intra", "extra", "myelin")                     # the DiSCo pack's pools, as its tissue mapping names them
B0_PRESETS = {"along z (the strands' frame)": (0.0, 0.0, 1.0), "transverse (x): 90° from z, as in a biplanar magnet like the Swoop": (1.0, 0.0, 0.0)}


def catalogue(field_T):
    """dmipy-sim's canonical white matter at the catalogue field nearest ``field_T`` (in log distance): ``T2`` and
    ``T1`` per pool (s), ``rho`` (m/s), ``chi_iso`` and ``chi_aniso`` (SI), and ``catalogue_field``, the field the
    numbers were cited at (the page says so when it is not the chosen one)."""
    import warnings
    from dmipy_sim.substrate.biophysical_constants import canonical_white_matter
    near = min(CATALOGUE_FIELDS, key=lambda f: abs(np.log(f / float(field_T))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = canonical_white_matter(field_T=near)
    return dict(catalogue_field=near, T2={q: float(w[f"T2_{q}"]) for q in POOLS}, T1={q: float(w[f"T1_{q}"]) for q in POOLS},
                rho=float(w["rho2"]), chi_iso=float(w["chi_iso_myelin"]), chi_aniso=float(w["delta_chi_a"]))


def b0_direction(theta_deg, phi_deg):
    """The unit field direction at polar angle ``theta`` from z and azimuth ``phi`` from x (degrees)."""
    t, f = np.radians(float(theta_deg)), np.radians(float(phi_deg))
    return (float(np.sin(t) * np.cos(f)), float(np.sin(t) * np.sin(f)), float(np.cos(t)))


@dataclass(frozen=True)
class Physics:
    """The tissue and the scanner a replay is evaluated at: the field (T) and its direction in the substrate frame,
    T2 and T1 per pool (s), the walls' surface relaxivity ``rho`` (m/s), myelin's ``chi_iso`` and ``chi_aniso``
    (SI); ``relaxation`` / ``contact`` / ``field`` switch the three tiers, so a tier is a knob the page can turn off
    one at a time. :meth:`tissue` is the :class:`dmipy_sim.spec.tissue.Tissue` for the replay (None when every tier
    is off: bare diffusion)."""
    field_T: float
    T2: dict
    T1: dict
    rho: float
    chi_iso: float
    chi_aniso: float
    b0_direction: tuple = (0.0, 0.0, 1.0)
    relaxation: bool = True
    contact: bool = True
    field: bool = True

    def __post_init__(self):
        if not (0 < self.field_T < 30):
            raise ValueError(f"the field is in tesla, got {self.field_T}")
        for what, m in (("T2", self.T2), ("T1", self.T1)):
            if set(m) != set(POOLS) or any(not (0 < float(v) < 100) for v in m.values()):
                raise ValueError(f"{what} is seconds per pool {POOLS}, got {m}")
        if self.rho < 0:
            raise ValueError("the surface relaxivity is non-negative")
        u = np.asarray(self.b0_direction, np.float64)
        if u.shape != (3,) or not np.isclose(np.linalg.norm(u), 1.0, atol=1e-6):
            raise ValueError("b0_direction is a unit vector")

    @classmethod
    def at(cls, field_T, *, b0_direction=(0.0, 0.0, 1.0), **overrides):
        """The catalogue's white matter at ``field_T`` with ``overrides`` (any field of the class)."""
        c = catalogue(field_T); c.pop("catalogue_field")
        return cls(field_T=float(field_T), b0_direction=tuple(float(x) for x in b0_direction), **{**c, **overrides})

    @property
    def bare(self):
        return not (self.relaxation or self.contact or self.field)

    def tissue(self):
        from dmipy_sim.spec.tissue import Tissue
        if self.bare:
            return None
        return Tissue(T2=dict(self.T2) if self.relaxation else None, T1=dict(self.T1) if self.relaxation else None,
                      rho=self.rho if self.contact else None, chi_iso=self.chi_iso if self.field else None,
                      chi_aniso=self.chi_aniso if self.field else 0.0)

    def label(self):
        if self.bare:
            return "bare diffusion"
        tiers = [n for n, on in (("relaxation", self.relaxation), ("contact", self.contact), ("field", self.field)) if on]
        u = self.b0_direction
        return f"{self.field_T:g} T along ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f}), tiers {'+'.join(tiers)}"


# ---- the stages -------------------------------------------------------------------------------------------------

def replay(layout, meas, physics=None):
    """``(dwi, floor, seconds)``: the S0-normalised signal of every voxel per measurement (NaN outside the pack's
    rows), the largest split-half floor over the timing classes, and the time; at ``physics`` (a :class:`Physics`:
    the tissue, the field and its direction), else bare diffusion."""
    t0 = time.perf_counter()
    S = np.full(layout.mask.shape + (len(meas.bvals),), np.nan)
    floor = np.zeros(layout.mask.shape)
    tissue = physics.tissue() if physics else None
    scanner = physics.field_T if tissue is not None else None
    b0 = physics.b0_direction if physics else (0.0, 0.0, 1.0)
    for name in np.unique(meas.shape):
        rows = meas.shape == name
        S[..., rows], f = layout.moments.image(name, meas.bvals[rows] * 1e6, meas.dirs[rows], backend=layout.backend,
                                                resident=layout.resident, tissue=tissue, scanner=scanner, b0_direction=b0)
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


def gradient_needed(b, delta, Delta):
    """The square-pulse PGSE amplitude (T/m) that gives ``b`` (s/mm^2) at ``delta`` / ``Delta`` (s):
    b = γ² G² δ² (Δ − δ/3)."""
    from dmipy_sim.acquisition.scanners import GAMMA
    return float(np.sqrt(b * 1e6 / (GAMMA ** 2 * delta ** 2 * (Delta - delta / 3.0))))


def scanner_classes():
    """The catalogue's scanner classes with their gradient limit and field: ``{name: (G_max T/m, field_T or None)}``."""
    from dmipy_sim.acquisition.scanners import ScannerLimits
    from dmipy_sim.acquisition import scanner_constants as scc
    out = {}
    for c in scc.SCANNER_CONSTANTS["classes"]:
        L = ScannerLimits.of(c)
        out[c] = (float(L.G_max), L.field_T)
    return out


def playable(protocol, shapes, scanner=None):
    """Per shell, the gradient it needs and whether ``scanner`` (a catalogue class, or None for no limit) can play
    it: rows of ``(shell, b, timing, G_needed T/m, G_max T/m or None, ok)``. A stimulated-echo class needs the same
    amplitude as its PGSE twin (the same δ and diffusion time)."""
    G_max = scanner_classes()[scanner][0] if scanner else None
    rows = []
    for s in protocol.shells:
        t = dict(delta=s.delta, Delta=s.Delta) if s.free_timing else shapes[s.shape]
        G = gradient_needed(s.b, float(t["delta"]), float(t["Delta"]))
        rows.append((s.shape, s.b, f"δ {float(t['delta']) * 1e3:g} / Δ {float(t['Delta']) * 1e3:g} ms", G, G_max, G_max is None or G <= G_max))
    return rows


def repeat_tracking(res, layout, tracking, keys):
    """The tractography of ``res`` (its FOD field) repeated over tracker ``keys``: a generator of ``(key, matrix,
    score, seconds)``, the replay, noise and CSD kept (only the tracker's randomness varies)."""
    from dataclasses import replace
    for k in keys:
        t0 = time.perf_counter()
        tg, _, _ = track(res.sh, layout, replace(tracking, key=int(k)))
        M = connectome(tg, layout)
        yield int(k), M, score(M, layout), time.perf_counter() - t0


def pair_spread(matrices, scores):
    """Over repeated runs: the streamline-count mean and standard deviation per region pair, the Pearson-vs-count
    mean and standard deviation, and the median coefficient of variation over the pairs that any run connected."""
    M = np.stack(matrices).astype(np.float64)
    mean = M.mean(0); std = M.std(0, ddof=1) if len(M) > 1 else np.zeros_like(mean)
    iu = np.triu_indices(N_REGIONS, 1)
    any_ = mean[iu] > 0
    cv = std[iu][any_] / mean[iu][any_]
    pc = np.array([s["pearson_count"] for s in scores])
    return dict(mean=mean, std=std, n=len(M), pearson_mean=float(pc.mean()), pearson_std=float(pc.std(ddof=1)) if len(pc) > 1 else 0.0,
                cv_median=float(np.median(cv)) if cv.size else float("nan"), pairs_any=int(any_.sum()),
                pairs_always=int((M[:, iu[0], iu[1]] > 0).all(0).sum()))


def sample_tractogram(tg, n, *, seed=0):
    """A :class:`dmipy_tract.Tractogram` of ``n`` streamlines drawn without replacement from ``tg`` (all of them
    when it has fewer), in their original order: what the page keeps and draws, while the full tractogram goes to
    the .tck file (disco-space#4 iteration 6)."""
    from dmipy_tract.tractogram import Tractogram
    if len(tg) <= n:
        return tg
    pick = np.sort(np.random.default_rng(seed).choice(len(tg), n, replace=False))
    counts = np.diff(tg.offsets)[pick]
    points = np.concatenate([tg.points[tg.offsets[i]:tg.offsets[i + 1]] for i in pick])
    return Tractogram(points, np.concatenate([[0], np.cumsum(counts)]), tg.seed_index[pick], tg.stop_reason[pick])


def retime(protocol, shape):
    """``protocol`` with every shell on the timing class ``shape`` (b-values, directions and counts kept): the A/B
    knob that swaps a PGSE class for its stimulated-echo twin, or one δ / Δ for another."""
    return Protocol(tuple(Shell(shape, s.b, s.n_dirs) for s in protocol.shells), n_b0=protocol.n_b0,
                    directions=protocol.directions, bvals=protocol.bvals, name=f"{protocol.name} on {shape}")


def compare(a, b):
    """A against B over the 120 region pairs: the Pearson between the two streamline-count matrices, the pairs
    connected in one only, and B's score minus A's."""
    iu = np.triu_indices(N_REGIONS, 1)
    ma, mb = a.matrix[iu], b.matrix[iu]
    return dict(pearson_ab=float(np.corrcoef(ma, mb)[0, 1]) if ma.std() > 0 and mb.std() > 0 else float("nan"),
                only_a=int(((ma > 0) & (mb == 0)).sum()), only_b=int(((mb > 0) & (ma == 0)).sum()),
                delta_count=b.score["pearson_count"] - a.score["pearson_count"], delta_area=b.score["pearson_area"] - a.score["pearson_area"])


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
    physics: Optional[Physics] = None


STAGES = ("replay", "noise", "csd", "track", "score")


def run_stages(layout, protocol, *, snr=None, tracking=Tracking(), noise_seed=0, physics=None):
    """The pipeline as a generator: before each of the ``STAGES`` it yields ``(stage, k, n)`` (so a page can show
    where the run is, from inside any worker), and last the :class:`Result`. ``physics`` is the tissue and scanner
    the replay is evaluated at (None: bare diffusion)."""
    t_run = time.perf_counter()

    def at(stage):
        print(f"[pipeline] {stage} at +{time.perf_counter() - t_run:.1f} s", flush=True)      # the server log shows where a run is
        return stage, STAGES.index(stage), len(STAGES)
    meas = measurements(protocol, layout.shapes)
    yield at("replay")
    if isinstance(layout, Columns):
        dwi, floor, t_replay = replay_full(layout, meas, physics)
    else:
        dwi, floor, t_replay = replay(layout, meas, physics)
    yield at("noise"); t0 = time.perf_counter(); noisy = add_noise(dwi, snr, seed=noise_seed, backend=layout.backend); t_noise = time.perf_counter() - t0
    signal = np.isfinite(dwi[..., 0])
    yield at("csd"); sh, t_csd = csd(noisy, meas, signal, backend=layout.backend)
    yield at("track"); tg, seeds, t_track = track(sh, layout, tracking)
    yield at("score"); t0 = time.perf_counter(); M = connectome(tg, layout); s = score(M, layout); t_score = time.perf_counter() - t0
    yield Result(protocol, meas, noisy, floor, sh, tg, seeds, M, s, snr=snr, physics=physics,
                 seconds=dict(replay=t_replay, noise=t_noise, csd=t_csd, track=t_track, score=t_score,
                              total=t_replay + t_noise + t_csd + t_track + t_score))


def run(layout, protocol, *, snr=None, tracking=Tracking(), noise_seed=0, physics=None, progress=None):
    """The whole pipeline for one protocol: every stage's output and time. ``progress(stage, k, n)`` is called as
    each stage begins."""
    for item in run_stages(layout, protocol, snr=snr, tracking=tracking, noise_seed=noise_seed, physics=physics):
        if isinstance(item, Result):
            return item
        if progress:
            progress(*item)
