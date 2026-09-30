"""The DiSCo Space's pipeline: an acquisition on a replay source -> the DWI of the 40^3 grid -> noise -> CSD ->
probabilistic tracking from the sixteen regions -> the 16 x 16 connectome against the dataset's ground truth. Plain
functions with their timings; nothing here draws or reads a widget.

Two sources replay the same walk. :class:`Layout` (demo mode) is dmipy-sim's shape-moment layout: the DiSCo replay
pack contracted once per stored pulse timing ("class"), so a protocol on those classes at any b-values, directions,
tissue and field is an elementwise kernel on device-resident rows, in seconds. :class:`Columns` (full mode) is the
columnar replay pack itself, read once per run for any pulse timing, in minutes. The CSD is dmipy-fit's batched
Tournier 2007 solver at order 8 on the response ``white_matter_response_tournier07`` estimates from the volume; the
tracker is dmipy-tract's ``track``; the score is the Pearson correlation of the symmetrised streamline counts with
the strand-count and cross-sectional-area matrices over the 120 region pairs.
"""
from __future__ import annotations

import os
import time
try:
    import tomllib                                   # 3.11+
except ModuleNotFoundError:                          # the ZeroGPU image is Python 3.10
    import tomli as tomllib
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(HERE, "config.toml")
DATA_DIR = os.path.join(ROOT, "data")
SH_ORDER = 8
N_REGIONS = 16
B0_MAX = 50.0                                        # s/mm^2: a row below it is a b = 0 measurement, everywhere in the pipeline
BAND_TOL = 0.005                                     # the band tolerance (x the pack's floor) of the layout and of a full replay
VOXEL_M = 25e-6                                      # DiSCo's voxel; the strands' coordinate unit
BACKENDS = ("jax", "torch")
MODES = ("demo", "full")


def config(path=CONFIG_PATH):
    with open(path, "rb") as f:
        return tomllib.load(f)


def mode(cfg):
    """``demo``: the shape-moment layout (seconds; the stored pulse timings). ``full``: the columnar replay pack
    (minutes per volume; any timing). ``DISCO_MODE`` in the environment, else ``[mode] mode``."""
    m = os.environ.get("DISCO_MODE") or cfg["mode"]["mode"]
    if m not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {m!r}")
    return m


def columns_uri(cfg):
    """Where full mode reads the columnar pack: ``DISCO_COLUMNS`` (a directory, or ``hf://owner/name/prefix``), else
    ``[mode] columns``."""
    return os.environ.get("DISCO_COLUMNS") or cfg["mode"]["columns"]


def resident(cfg):
    """Whether the layout's device tensors stay across calls: ``DISCO_RESIDENT`` (``0``/``false`` for a pool that
    drops the device between calls) else ``[compute] resident``."""
    env = os.environ.get("DISCO_RESIDENT")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no")
    return bool(cfg["compute"]["resident"])


def backend(cfg):
    """The compute backend, ``jax`` or ``torch`` (a host that runs PyTorch only: the layout image, the CSD solver
    and the tracker switch together): ``DISCO_BACKEND`` in the environment, else ``[compute] backend``."""
    b = os.environ.get("DISCO_BACKEND") or cfg["compute"]["backend"]
    if b not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {b!r}")
    return b


# ---- the acquisition -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shell:
    """One shell: the timing class it plays, its b-value (s/mm^2) and its direction count. A shell made by
    :meth:`free` carries its own pulse timing (``delta``, ``Delta``, ``TE`` in seconds) instead of a stored class,
    and ``shape`` names it."""
    shape: str
    b: float
    n_dirs: int
    delta: Optional[float] = None
    Delta: Optional[float] = None
    TE: Optional[float] = None

    @classmethod
    def free(cls, b, n_dirs, delta, Delta, TE):
        """A shell at its own square-pulse timing (seconds), named ``d<delta>-D<Delta>-TE<TE>`` in ms."""
        if not (0 < delta < Delta < TE):
            raise ValueError(f"a free pulse timing needs 0 < delta < Delta < TE, got {delta}, {Delta}, {TE}")
        return cls(f"d{delta * 1e3:g}-D{Delta * 1e3:g}-TE{TE * 1e3:g}", float(b), int(n_dirs), float(delta), float(Delta), float(TE))

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
        return self.bvals < B0_MAX


def measurements(protocol, shapes):
    """The rows of ``protocol`` with the timings of ``shapes`` (the config's ``[shapes]`` table) or the shells' own."""
    timing = {}
    for s in protocol.shells:
        if s.free_timing:
            timing[s.shape] = dict(delta=s.delta, Delta=s.Delta, TE=s.TE)
        elif s.shape not in shapes:
            raise KeyError(f"no timing class {s.shape!r}; the layout holds {sorted(shapes)}")
        else:
            timing[s.shape] = shapes[s.shape]
    names = [protocol.shells[0].shape] * protocol.n_b0
    for s in protocol.shells:
        names += [s.shape] * s.n_dirs
    if protocol.directions is None:
        from dmipy_tract import hemisphere
        dirs = np.concatenate([np.tile([0.0, 0.0, 1.0], (protocol.n_b0, 1))] + [hemisphere(s.n_dirs) for s in protocol.shells])
    else:
        dirs = np.asarray(protocol.directions, np.float64)
    if protocol.bvals is None:
        b = np.asarray([0.0] * protocol.n_b0 + [s.b for s in protocol.shells for _ in range(s.n_dirs)], np.float64)
    else:
        b = np.asarray(protocol.bvals, np.float64)
    delta = np.array([timing[n]["delta"] for n in names]); Delta = np.array([timing[n]["Delta"] for n in names])
    TE = np.array([timing[n]["TE"] for n in names])
    return Measurements(b, dirs, np.asarray(names), delta, Delta, TE)


def protocol_from_rows(bvals, dirs, group, make_shell, *, name):
    """A Protocol from a table of rows (``bvals`` in s/mm^2, ``dirs`` (N, 3) in any norm, ``group`` an integer per
    row saying which shell it belongs to) and ``make_shell(group, rows) -> Shell``: the b = 0 rows first (their
    direction set to z), then every group's rows in ascending group order, each row's own b-value and unit
    direction. Returns the protocol and ``idx``, the table rows in the protocol's order."""
    bvals = np.asarray(bvals, np.float64).ravel(); dirs = np.asarray(dirs, np.float64); group = np.asarray(group)
    if dirs.ndim != 2 or dirs.shape != (len(bvals), 3):
        raise ValueError(f"dirs must be ({len(bvals)}, 3), got {dirs.shape}")
    if np.any(bvals < 0) or not np.all(np.isfinite(bvals)) or not np.all(np.isfinite(dirs)):
        raise ValueError("the table has a negative or non-finite entry")
    b0 = bvals < B0_MAX
    if not b0.any():
        raise ValueError("the table needs a b = 0 row (the DWI is normalised by it)")
    norm = np.linalg.norm(dirs, axis=1)
    if np.any(norm[~b0] == 0):
        raise ValueError("a b > 0 row has a zero direction")
    dirs = np.where(b0[:, None], np.array([0.0, 0.0, 1.0]), dirs / np.where(norm > 0, norm, 1.0)[:, None])
    idx = [np.flatnonzero(b0)]; shells = []
    for g in np.unique(group[~b0]):
        rows = np.flatnonzero((group == g) & ~b0)
        idx.append(rows); shells.append(make_shell(g, rows))
    idx = np.concatenate(idx)
    return Protocol(tuple(shells), n_b0=int(b0.sum()), directions=dirs[idx], bvals=bvals[idx], name=name), idx


def protocol_from_scheme(path, *, name="uploaded scheme"):
    """A Camino ``STEJSKALTANNER`` scheme file (``gx gy gz |G| Delta delta TE`` per row, SI units) as a Protocol
    with every row's own b-value, direction and pulse timing, for full mode: one free shell per distinct
    (b, delta, Delta, TE) in that order, the b = 0 rows first. Read by :func:`dmipy_sim.io.mcdc.read_scheme`, the
    one reader of that format."""
    from dmipy_sim.io.mcdc import read_scheme
    enc = read_scheme(path).encoding
    b = np.asarray(enc.bvalues, np.float64) / 1e6
    delta = np.asarray(enc.delta, np.float64); Delta = np.asarray(enc.Delta, np.float64); TE = np.asarray(enc.TE, np.float64)
    keys = np.stack([np.round(b), np.round(delta * 1e6), np.round(Delta * 1e6), np.round(TE * 1e6)], 1)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    def make_shell(g, rows):
        r = rows[0]
        return Shell.free(keys[r, 0], len(rows), delta[r], Delta[r], TE[r])
    return protocol_from_rows(b, np.asarray(enc.gradient_directions, np.float64), group.ravel(), make_shell, name=name)[0]


def disco_protocol(cfg):
    """DiSCo's own 364-measurement protocol (its gradient table, its per-shell timing classes from ``[disco]``) as
    a Protocol with every row's own direction and b-value: the acquisition the reference volumes were replayed with.
    Returns it and the table rows in the protocol's order (the reference volumes follow the table's)."""
    bv = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_gradients.bvals")).ravel()
    dirs = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_gradients_dipy.bvecs"))
    if dirs.shape[0] == 3:
        dirs = dirs.T
    shells = cfg["disco"]["shells"]                                     # [{b, shape}] in the table's row order
    centres = np.array([s["b"] for s in shells])
    which = np.argmin(np.abs(bv[:, None] - centres[None, :]), axis=1)
    return protocol_from_rows(bv, dirs, which, lambda k, rows: Shell(shells[k]["shape"], float(np.round(bv[rows].mean())), int(len(rows))),
                              name="DiSCo 364")


def preset_protocol(cfg, name):
    """A ``[presets]`` entry of the config as a Protocol."""
    p = cfg["presets"][name]
    return Protocol(tuple(Shell(s["shape"], float(s["b"]), int(s["n_dirs"])) for s in p["shells"]), n_b0=int(p["n_b0"]), name=name)


def retime(protocol, shape):
    """``protocol`` with every shell on the timing class ``shape`` (b-values, directions and counts kept): the A/B
    knob that swaps a PGSE class for its stimulated-echo twin, or one δ / Δ for another."""
    return Protocol(tuple(Shell(shape, s.b, s.n_dirs) for s in protocol.shells), n_b0=protocol.n_b0,
                    directions=protocol.directions, bvals=protocol.bvals, name=f"{protocol.name} on {shape}")


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


# ---- the physics: tissue and scanner --------------------------------------------------------------------------------
CATALOGUE_FIELDS = (1.5, 3.0, 7.0)                       # where dmipy-sim's white-matter catalogue has cited relaxation
CATALOGUE_POOLS = ("intra", "extra", "myelin")           # the pools the catalogue has T2 and T1 for
B0_ALONG_Z = (0.0, 0.0, 1.0)
B0_TRANSVERSE = (1.0, 0.0, 0.0)


def catalogue(field_T, pools=CATALOGUE_POOLS):
    """dmipy-sim's canonical white matter at the catalogue field nearest ``field_T`` (in log distance): ``T2`` and
    ``T1`` per pool of ``pools`` (s), ``rho`` (m/s), the sheath's ``chi_iso`` and ``chi_aniso`` (SI), and
    ``catalogue_field``, the field the numbers were cited at (the page says so when it is not the chosen one)."""
    import warnings
    from dmipy_sim.substrate.biophysical_constants import canonical_white_matter
    if not float(field_T) > 0:
        raise ValueError(f"the field is in tesla, got {field_T}")
    near = min(CATALOGUE_FIELDS, key=lambda f: abs(np.log(f / float(field_T))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = canonical_white_matter(field_T=near)
    unknown = [q for q in pools if q not in CATALOGUE_POOLS]
    if unknown:
        raise ValueError(f"the catalogue has no relaxation for the pools {unknown}; it knows {CATALOGUE_POOLS}")
    return dict(catalogue_field=near, T2={q: float(w[f"T2_{q}"]) for q in pools}, T1={q: float(w[f"T1_{q}"]) for q in pools},
                rho=float(w["rho2"]), chi_iso=float(w["chi_iso_myelin"]), chi_aniso=float(w["delta_chi_a"]))


def b0_direction(theta_deg, phi_deg):
    """The unit field direction at polar angle ``theta`` from z and azimuth ``phi`` from x (degrees)."""
    t, f = np.radians(float(theta_deg)), np.radians(float(phi_deg))
    return (float(np.sin(t) * np.cos(f)), float(np.sin(t) * np.sin(f)), float(np.cos(t)))


@dataclass(frozen=True)
class Physics:
    """The tissue and the scanner a replay is evaluated at: the field (T) and its direction in the substrate frame,
    T2 and T1 per seeded pool (s), the walls' surface relaxivity ``rho`` (m/s), the sheath's ``chi_iso`` and
    ``chi_aniso`` (SI, the field source); ``relaxation`` / ``contact`` / ``field`` switch the three tiers, so a tier
    is a knob the page can turn off one at a time. :meth:`tissue` is the :class:`dmipy_sim.spec.tissue.Tissue` for
    the replay (None when every tier is off: bare diffusion)."""
    field_T: float
    T2: dict
    T1: dict
    rho: float
    chi_iso: float
    chi_aniso: float
    b0_direction: tuple = B0_ALONG_Z
    relaxation: bool = True
    contact: bool = True
    field: bool = True

    def __post_init__(self):
        if not (0 < self.field_T < 30):
            raise ValueError(f"the field is in tesla, got {self.field_T}")
        if not self.T2 or set(self.T1) != set(self.T2):
            raise ValueError(f"T2 and T1 are seconds over the same pools, got {self.T2} and {self.T1}")
        for what, m in (("T2", self.T2), ("T1", self.T1)):
            if any(not (0 < float(v) < 100) for v in m.values()):
                raise ValueError(f"{what} is seconds per pool, got {m}")
        if self.rho < 0:
            raise ValueError("the surface relaxivity is non-negative")
        u = np.asarray(self.b0_direction, np.float64)
        if u.shape != (3,) or not np.isclose(np.linalg.norm(u), 1.0, atol=1e-6):
            raise ValueError("b0_direction is a unit vector")

    @classmethod
    def at(cls, field_T, *, pools=("intra", "extra"), b0_direction=B0_ALONG_Z, **overrides):
        """The catalogue's white matter at ``field_T`` over ``pools`` with ``overrides`` (any field of the class)."""
        c = catalogue(field_T, pools); c.pop("catalogue_field")
        return cls(field_T=float(field_T), b0_direction=tuple(float(x) for x in b0_direction), **{**c, **overrides})

    @property
    def bare(self):
        return not (self.relaxation or self.contact or self.field)

    @property
    def along_z(self):
        return bool(np.allclose(self.b0_direction, B0_ALONG_Z))

    @property
    def pools(self):
        return tuple(self.T2)

    def tissue(self, unseeded=()):
        """The Tissue for the replay; ``unseeded`` names the substrate's pools no walker was seeded in, which the
        library's mapping still wants a value for: they take the first seeded pool's (nothing reads it)."""
        from dmipy_sim.spec.tissue import Tissue
        if self.bare:
            return None
        first = self.pools[0]
        T2 = {**self.T2, **{q: self.T2[first] for q in unseeded}}; T1 = {**self.T1, **{q: self.T1[first] for q in unseeded}}
        return Tissue(T2=T2 if self.relaxation else None, T1=T1 if self.relaxation else None,
                      rho=self.rho if self.contact else None, chi_iso=self.chi_iso if self.field else None,
                      chi_aniso=self.chi_aniso if self.field else 0.0)

    def label(self):
        if self.bare:
            return "bare diffusion"
        tiers = [n for n, on in (("relaxation", self.relaxation), ("contact", self.contact), ("field", self.field)) if on]
        u = self.b0_direction
        return f"{self.field_T:g} T along ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f}), tiers {'+'.join(tiers)}"


def tissue_and_scanner(physics, unseeded=()):
    """``(tissue, scanner)`` for a replay at ``physics``: ``(None, None)`` for bare diffusion; the scanner's field
    goes with the field tier only (relaxation at a field is already in the tissue's T2 and T1)."""
    tissue = physics.tissue(unseeded) if physics else None
    return tissue, (physics.field_T if tissue is not None and physics.field else None)


# ---- the sources: what replays the walk -------------------------------------------------------------------------------

class Source:
    """What a replay source has beside its data: the DiSCo mask, the sixteen regions, the two ground-truth
    matrices, the config's timing classes, the compute backend, and ``mode`` (``demo`` or ``full``). A source
    validates a run, replays measurements, plans a replay and reports its accuracy."""
    mode = ""

    unseeded = ()                                    # the substrate's pools no walker was seeded in

    def __init__(self, cfg):
        import nibabel as nib
        self.cfg = cfg
        self.shapes = cfg["shapes"]
        self.pools = tuple(cfg["physics"]["pools"])
        self.backend = backend(cfg)
        self.mask = np.asarray(nib.load(os.path.join(DATA_DIR, "DiSCo_mask.nii.gz")).dataobj) > 0
        self.rois = np.asarray(nib.load(os.path.join(DATA_DIR, "DiSCo_ROIs.nii.gz")).dataobj).astype(np.int32)
        self.gt_count = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_Connectivity_Matrix_Strands_Count.txt"))
        self.gt_area = np.loadtxt(os.path.join(DATA_DIR, "DiSCo_Connectivity_Matrix_Cross-Sectional_Area.txt"))

    def check_grid(self, grid_shape, what):
        if tuple(grid_shape) != self.mask.shape:
            raise ValueError(f"{what}'s grid {tuple(grid_shape)} is not the mask's {self.mask.shape}")

    def validate(self, protocol, physics):
        """Refuses, before any work, a run this source cannot do; returns its measurements."""
        return measurements(protocol, self.shapes)

    def replay(self, meas, physics=None):
        """``(dwi, floor, s0_factor, seconds)``: the S0-normalised signal of every voxel per measurement (NaN
        outside the pack's rows), the split-half floor per voxel, the voxel's raw b = 0 signal as a fraction of M0
        (the bare signal of the fullest water voxel: what relaxation, contact and the field took, 1 when the source
        cannot tell), and the time; at ``physics`` else bare diffusion."""
        raise NotImplementedError

    def plan(self, meas, physics=None):
        """What a replay reads before any byte moves (None when the source has nothing to plan)."""
        return None

    def warm(self):
        """Whatever the first run would otherwise pay (nothing for a source that reads per run)."""
        return

    def release(self):
        """Whatever a run kept on the device that must not survive it (nothing for a source that reads per run)."""
        return

    def ingredients(self, meas, physics=None):
        """The per-voxel maps of what each tier multiplies into the sum (None for a source without them)."""
        return None

    def ladder(self, meas, physics):
        """The noise-free replays with the tiers switched on one at a time, ``[(label, dwi)]`` from bare diffusion
        up to (not including) ``physics`` itself; empty for a source that cannot afford them."""
        return []

    def accuracy(self, res=None):
        """Rows ``(what, value)`` saying what this source's replay is an approximation of, and ``res``'s own floor."""
        rows = []
        if res is not None:
            f = floor_stats(res)
            rows.append(["this run's floor: median / 99 % / max over voxels with signal", f"{f['median']:.4f} / {f['p99']:.4f} / {f['max']:.4f}"])
            sf = res.s0_factor[np.isfinite(res.dwi[..., 0])]
            rows.append(["this run's b = 0 signal over M0: min / median / max", f"{np.nanmin(sf):.3f} / {np.nanmedian(sf):.3f} / {np.nanmax(sf):.3f}"])
            snr = b0_snr(res)
            if snr:
                rows.append([f"b = 0 SNR at SNR {res.snr:g} at M0: min / median / max", f"{snr['min']:.1f} / {snr['median']:.1f} / {snr['max']:.1f}"])
        return rows


def s0_normalised(S, b0, M0=None):
    """``(S / S0, S0 / M0_max)``: every voxel's signal over its own b = 0 mean, and that b = 0 mean over the largest
    M0 (the bare signal of the fullest water voxel: ``M0`` is the bare b = 0 map, None gives a factor of 1)."""
    S0 = np.nanmean(S[..., b0], axis=-1)
    factor = S0 / np.nanmax(M0) if M0 is not None else np.where(np.isfinite(S0), 1.0, np.nan)
    return S / S0[..., None], factor


class Layout(Source):
    """Demo mode: dmipy-sim's shape-moment layout (``dmipy_sim.replay.shape_moments``) at the config's dataset
    revision, or a local copy; every stored class's timing is the config's, and a tissue needs the layout's tiers."""
    mode = "demo"

    def __init__(self, cfg, *, local=None):
        from dmipy_sim.replay.shape_moments import ShapeMoments
        super().__init__(cfg)
        d = cfg["data"]
        uri = local or f"hf://{d['repo']}/{d['moments']}"
        self.moments = ShapeMoments.open(uri, revision=d.get("revision") or None) if uri.startswith("hf://") else ShapeMoments(uri)
        self.resident = resident(cfg)
        want = d.get("source_manifest_sha256")
        got = self.moments.manifest["source"]["manifest_sha256"]
        if want and got != want:
            raise ValueError(f"the layout at {uri} was contracted from columnar manifest {got[:12]}, the config pins {want[:12]}")
        missing = [s for s in self.shapes if s not in self.moments.shapes]
        if missing:
            raise ValueError(f"the layout at {uri} lacks the timing classes {missing}; it holds {self.moments.shapes}")
        for name, t in self.shapes.items():
            enc = self.moments.manifest["shapes"][name].get("encoding") or {}
            for k in ("delta", "Delta", "TE"):
                v = enc.get(k)
                v = v[0] if isinstance(v, list) and v else v
                if v is not None and abs(float(v) - float(t[k])) > 1e-9:
                    raise ValueError(f"class {name}: the layout's {k} is {v} s, the config's {t[k]} s")
        self.tiers = bool(self.moments.manifest.get("tiers"))
        self.M0 = None                                   # the bare b = 0 map (every voxel's walker weight): set by warm()
        if self.tiers:                                   # the pools with walkers are the spec's first n_pools ids
            t = self.moments.manifest["tiers"]
            spec_pools = sorted(t["substrate"]["pools"], key=lambda q: q["id"]) if t.get("substrate") else []
            seeded = tuple(q["name"] for q in spec_pools[:int(t["pools"])])
            if set(seeded) != set(self.pools):
                raise ValueError(f"the layout's seeded pools are {seeded}, the config's [physics] pools {self.pools}")
            self.unseeded = tuple(q["name"] for q in spec_pools[int(t["pools"]):])
        self.check_grid(self.moments.grid.shape, "the layout")

    def validate(self, protocol, physics):
        meas = measurements(protocol, self.shapes)
        if physics and not physics.bare and not self.tiers:
            raise ValueError("this layout holds bare diffusion only: switch the tissue panel off")
        if physics and set(physics.pools) != set(self.pools):
            raise ValueError(f"the tissue names the pools {physics.pools}; this layout's are {self.pools}")
        return meas

    def replay(self, meas, physics=None):
        t0 = time.perf_counter()
        S = np.full(self.mask.shape + (len(meas.bvals),), np.nan)
        floor = np.zeros(self.mask.shape)
        tissue, scanner = tissue_and_scanner(physics, self.unseeded)
        b0 = physics.b0_direction if physics else B0_ALONG_Z
        for name in np.unique(meas.shape):
            rows = meas.shape == name
            S[..., rows], f = self.moments.image(name, meas.bvals[rows] * 1e6, meas.dirs[rows], backend=self.backend,
                                                 resident=True, tissue=tissue, scanner=scanner, b0_direction=b0)
            floor = np.fmax(floor, f)                    # resident within a run: release() drops the device copies after it
        if self.M0 is None:                              # the bare b = 0 map: the walker weight of every voxel
            self.M0 = self.moments.image(meas.shape[0], np.zeros(1), np.array([B0_ALONG_Z]), backend=self.backend, resident=True)[0][..., 0]
        dwi, factor = s0_normalised(S, meas.b0, self.M0)
        return dwi, floor, factor, time.perf_counter() - t0

    def release(self):
        if not self.resident:
            self.moments.release()

    def ingredients(self, meas, physics=None):
        """What each tier multiplies into the sum, per voxel, for the run's first class (the layout's
        :meth:`~dmipy_sim.replay.shape_moments.ShapeMoments.tier_maps` on the device columns the replay left
        resident): ``intra_fraction`` (the walker weight in the intra-axonal pool over the voxel's),
        ``wall_contact_um`` (the walkers' boundary local time l under the class's gate, a length; the layout stores
        it signed as the exponent's term, -l, so the contact tier's weight is exp(rho c / D) = exp(-rho l / D)),
        ``contact_survival`` (that factor at the run's rho, None without the contact tier), ``field_rad`` (the spread
        over the voxel's walkers of the dephasing phase the sheath's field gives them by the echo at the run's field
        and direction, the exact per-walker phase the kernel applies, in radians; None without the field tier), and
        ``D_walk`` (m^2/s)."""
        if not self.tiers:
            return None
        tissue, scanner = tissue_and_scanner(physics, self.unseeded)
        b0 = physics.b0_direction if physics else B0_ALONG_Z
        maps = self.moments.tier_maps(str(meas.shape[0]), tissue, scanner, backend=self.backend, resident=True, b0_direction=b0)
        contact = maps["contact"]
        return dict(D_walk=float(self.moments.manifest["tiers"]["D_walk"]), intra_fraction=maps["pool"]["intra"],
                    wall_contact_um=None if contact is None else -contact * 1e6,
                    contact_survival=maps["contact_weight"] if (physics and physics.contact) else None,
                    field_rad=maps["phase_std"] if (physics and physics.field) else None)

    def ladder(self, meas, physics):
        if physics is None or physics.bare:
            return []
        steps = [("bare diffusion", None)]
        on = []
        for tier in ("relaxation", "contact", "field"):
            if getattr(physics, tier):
                on.append(tier)
                if len(on) < sum(getattr(physics, q) for q in ("relaxation", "contact", "field")):
                    steps.append(("+ " + " + ".join(on), replace(physics, **{q: q in on for q in ("relaxation", "contact", "field")})))
        return [(label, self.replay(meas, ph)[0]) for label, ph in steps]

    def default_physics(self):
        """The tissue panel's default (``[physics]``), or None when it starts off or the layout has no tiers."""
        p = self.cfg["physics"]
        return Physics.at(float(p["default_field"]), pools=self.pools) if (p["default_on"] and self.tiers) else None

    def warm(self):
        """DiSCo's own protocol replayed once at the default physics (every class it plays compiled at the row
        counts the default run uses), then one direction on every other class (its tier group's terms contracted
        and on the device). When the tiles are not kept resident (a pool that drops the device between calls), the
        padded host arrays are loaded into this process instead, so a forked worker inherits them and pays the
        transfer alone."""
        if not self.resident:
            self.moments.preload(list(self.shapes))
            return
        physics = self.default_physics()
        disco = disco_protocol(self.cfg)[0]
        self.replay(measurements(disco, self.shapes), physics)
        rest = [name for name in self.shapes if name not in {s.shape for s in disco.shells}]
        if rest:
            self.replay(measurements(Protocol(tuple(Shell(name, 1000.0, 1) for name in rest), n_b0=1, name="warm"), self.shapes), physics)

    def accuracy(self, res=None):
        m = self.moments.manifest; src = m["source"]
        rows = [["temporal bands kept (K)", str(m["K"])],
                ["band error at the built amplitude (worst class)", f"{m['band_error']:.2e}"],
                ["band tolerance (× the pack's floor)", f"{m['tol']:g}"],
                ["source pack", str(src.get("pack"))],
                ["source pack's certified median floor", f"{src['floor']:.4g}"],
                ["source manifest sha256", src["manifest_sha256"][:16]],
                ["layout written by", f"{m.get('code', {}).get('commit', '?')[:12]} on {m.get('created', '?')}"],
                ["pulses", "square (slew rate ∞), one TE per class"],
                ["tiers stored", "pool, contact, field" if self.tiers else "none (bare diffusion only)"]]
        rows += super().accuracy(res)
        if res is not None:
            for name in np.unique(res.meas.shape):
                sh = m["shapes"][name]; enc = sh.get("encoding") or {}
                first = {k: (v[0] if isinstance(v, list) and v else v) for k, v in enc.items()}
                rows.append([f"class {name}", f"δ {first.get('delta')} / Δ {first.get('Delta')} / TE {first.get('TE')} s, "
                                              f"built at {float(sh.get('amplitude_built') or 0) * 1e3:.0f} mT/m, pathway amplitude {sh.get('pathway') or 1}"])
        return rows


class Columns(Source):
    """Full mode: the columnar replay pack (:class:`dmipy_sim.replay.columnar.ColumnarPack`) at
    :func:`columns_uri`, read once per run: one ScannerSequence per run (one echo time, one pulse kind, square
    pulses), the tissue and field of the run, the field along the pack's z."""
    mode = "full"
    RATE = {True: 45e6, False: 500e6}                # bytes/s read from the Hub, from a local disk

    def __init__(self, cfg, *, uri=None):
        from dmipy_sim.replay.columnar import ColumnarPack
        super().__init__(cfg)
        self.uri = uri or columns_uri(cfg)
        self.pack = ColumnarPack(self.uri, workers=int(cfg["mode"]["workers"]))
        self.check_grid(self.pack.grid.shape, "the pack")
        self.remote = self.uri.startswith("hf://")

    def kinds(self, meas):
        """The pulse kinds (``pgse`` / ``pgste``) of the classes the rows play; a free timing is PGSE."""
        return sorted({self.shapes[n].get("kind", "pgse") if n in self.shapes else "pgse" for n in np.unique(meas.shape)})

    def validate(self, protocol, physics):
        meas = measurements(protocol, self.shapes)
        if len(np.unique(np.round(meas.TE, 9))) != 1:
            raise ValueError("full mode replays one echo time per run")
        if len(self.kinds(meas)) != 1:
            raise ValueError("full mode replays one pulse kind (PGSE or stimulated echo) per run")
        if physics and not physics.along_z:
            raise ValueError("full mode replays the field along z only")
        return meas

    def sequence(self, meas):
        """The one ScannerSequence of the measurements: square pulses, every row's own delta / Delta at the rows'
        common TE; a stimulated echo stores for TM = Delta - delta between 90° pulses."""
        import dmipy_sim as d
        TE = float(np.unique(np.round(meas.TE, 9))[0])
        kind, = self.kinds(meas)
        if kind == "pgste":
            return d.pgste(meas.dirs.tolist(), meas.delta, meas.Delta - meas.delta, bvalues=meas.bvals * 1e6, n_t=1000,
                           slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
        return d.pgse(meas.dirs.tolist(), meas.delta, meas.Delta, bvalues=meas.bvals * 1e6, TE=TE, n_t=1000, slew_rate=np.inf)

    def plan(self, meas, physics=None):
        """What the replay will read (the pack's plan: bands, modes, tiers, rows, bytes) and ``estimated_seconds``
        at the source's read rate."""
        tissue, scanner = tissue_and_scanner(physics, self.unseeded)
        plan = self.pack.plan(self.sequence(meas), tissue=tissue, scanner=scanner, tol=BAND_TOL)
        plan["estimated_seconds"] = plan["bytes"] / self.RATE[self.remote]
        return plan

    def replay(self, meas, physics=None, *, progress=None):
        from dmipy_sim.replay.study import Study, Protocol as SProtocol, Acquisition
        t0 = time.perf_counter()
        tissue, scanner = tissue_and_scanner(physics, self.unseeded)
        study = Study(SProtocol([Acquisition(self.sequence(meas), name="run")]), tissues=[tissue], scanners=[scanner])
        S, floor, plan = self.pack.image_study(study, tol=BAND_TOL, chunk_rows=1_000_000, progress=progress)
        self.last_plan = plan
        dwi, factor = s0_normalised(S[0], meas.b0)      # the pack's bare weights are not read here: the factor is 1
        return dwi, floor[0], factor, time.perf_counter() - t0

    def accuracy(self, res=None):
        m = self.pack.meta
        rows = [["source pack", str(m.get("id"))], ["source pack's certified median floor", f"{self.pack.floor:.4g}"],
                ["bands kept", str(self.pack.K)], ["band tolerance (× the pack's floor)", f"{BAND_TOL:g}"],
                ["pulses", "square (slew rate ∞), one TE and one pulse kind per run"]]
        plan = getattr(self, "last_plan", None)
        if plan:
            rows.append(["last replay read", f"{plan.get('rows', 0):,} rows, {plan.get('bytes', 0) / 1e9:.1f} GB, bands {plan.get('K')}, modes {plan.get('modes')}"])
        return rows + super().accuracy(res)


def source(cfg, *, local=None):
    """The replay source the config's mode names: a :class:`Layout` (``DISCO_MOMENTS`` or ``local`` for a local
    copy) or a :class:`Columns`."""
    return Columns(cfg) if mode(cfg) == "full" else Layout(cfg, local=local or os.environ.get("DISCO_MOMENTS"))


# ---- the stages -------------------------------------------------------------------------------------------------

def add_noise(dwi, snr, s0_factor=None, seed=0, backend="jax"):
    """Rician noise at ``snr`` defined at M0, the bare signal of the fullest water voxel: sigma is M0 / snr in
    absolute units, so on the S0-normalised DWI it is ``1 / (snr * s0_factor)`` per voxel (``s0_factor`` the voxel's
    raw b = 0 over M0; None means 1 everywhere, the b = 0 SNR itself). ``None`` for ``snr`` leaves the signal
    noiseless. NaN voxels stay NaN. The draw comes from JAX's generator on the jax backend and from numpy's on the
    torch backend (a forked GPU worker must not touch JAX); the two streams differ, the distribution is the same."""
    if snr is None:
        return dwi
    if snr <= 0:
        raise ValueError("SNR is positive, or None for no noise")
    from dmipy_sim.acquisition.noise import add_rician_noise
    valid = np.isfinite(dwi)
    factor = np.ones(dwi.shape[:-1]) if s0_factor is None else np.asarray(s0_factor, np.float64)
    if factor.shape != dwi.shape[:-1] or np.any(factor[np.isfinite(factor)] <= 0):
        raise ValueError("s0_factor is a positive map over the DWI's voxels")
    sigma = np.broadcast_to((1.0 / (snr * factor))[..., None], dwi.shape)[valid]
    out = np.array(dwi)                                  # Rice(nu, sigma) = sigma * Rice(nu / sigma, 1): one unit-sigma draw
    out[valid] = sigma * np.asarray(add_rician_noise(dwi[valid] / sigma, 1.0, seed=seed, rng="numpy" if backend == "torch" else "jax"))
    return out


def scheme(meas):
    """The measurements as dmipy-fit's acquisition scheme (SI b-values, the same b = 0 threshold as the pipeline)."""
    from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
    return acquisition_scheme_from_bvalues(meas.bvals * 1e6, meas.dirs, delta=meas.delta, Delta=meas.Delta, TE=meas.TE, b0_threshold=B0_MAX * 1e6)


def csd(dwi, meas, mask, backend="jax"):
    """``(sh, seconds)``: the FOD field ``(X, Y, Z, 45)`` in the tournier07 basis, dmipy-fit's batched Tournier 2007
    solver on ``backend`` with the response ``white_matter_response_tournier07`` estimates from the masked voxels,
    fitted on ``mask`` (the voxels with signal)."""
    from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
    from dmipy_fit.tissue_response.white_matter_response import white_matter_response_tournier07
    t0 = time.perf_counter()
    data = np.nan_to_num(dwi, nan=0.0)
    sch = scheme(meas)
    _, response, _ = white_matter_response_tournier07(sch, data[mask])
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


def tracking_inputs(sh, source, density):
    """``(field, seeds)``: the FOD field on the mask and the ``density^3`` seeds per region voxel, built once for
    however many tracker keys run on them."""
    from dmipy_tract import FODField, seeds_from_mask
    return FODField(sh, np.eye(4), source.mask | (source.rois > 0)), seeds_from_mask(source.rois > 0, np.eye(4), density=density)


def track(field, seeds, settings, backend):
    """``(tractogram, seconds)``: probabilistic streamlines from ``seeds`` on ``field``."""
    from dmipy_tract import track as _track
    t0 = time.perf_counter()
    tg = _track(field, seeds, rule="probabilistic", step_mm=settings.step_mm, max_angle=settings.max_angle,
                max_steps=settings.max_steps, relative_threshold=settings.relative_threshold, key=settings.key, backend=backend)
    return tg, time.perf_counter() - t0


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


def connectome(tg, source):
    """The symmetrised 16 x 16 streamline-count matrix (no self-connections)."""
    from dmipy_tract import connectivity
    matrix, _ = connectivity(tg, source.rois, np.eye(4))
    M = matrix[1:N_REGIONS + 1, 1:N_REGIONS + 1].astype(np.float64)
    M = M + M.T
    np.fill_diagonal(M, 0)
    return M


PAIRS = np.triu_indices(N_REGIONS, 1)                # the 120 region pairs


def pearson(a, b):
    """The Pearson correlation of two vectors, NaN when either is constant."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def connected_pairs(M):
    """How many of the 120 region pairs ``M`` connects."""
    return int((M[PAIRS] > 0).sum())


def score(M, source):
    """Pearson correlations over the 120 region pairs with the strand-count and area matrices, and the pair
    bookkeeping (connected, ground-truth, false, missed)."""
    gt = source.gt_count
    return dict(pearson_count=pearson(M[PAIRS], gt[PAIRS]), pearson_area=pearson(M[PAIRS], source.gt_area[PAIRS]),
                connected_pairs=connected_pairs(M), gt_pairs=connected_pairs(gt),
                false_pairs=int(((M[PAIRS] > 0) & (gt[PAIRS] == 0)).sum()), missed_pairs=int(((M[PAIRS] == 0) & (gt[PAIRS] > 0)).sum()))


def compare(a, b):
    """A against B over the 120 region pairs: the Pearson between the two streamline-count matrices, the pairs
    connected in one only, and B's score minus A's."""
    ma, mb = a.matrix[PAIRS], b.matrix[PAIRS]
    return dict(pearson_ab=pearson(ma, mb), only_a=int(((ma > 0) & (mb == 0)).sum()), only_b=int(((mb > 0) & (ma == 0)).sum()),
                delta_count=b.score["pearson_count"] - a.score["pearson_count"], delta_area=b.score["pearson_area"] - a.score["pearson_area"])


def pair_spread(matrices, scores):
    """Over repeated runs: the streamline-count mean and standard deviation per region pair, the Pearson-vs-count
    mean and standard deviation, and the median coefficient of variation over the pairs that any run connected."""
    M = np.stack(matrices).astype(np.float64)
    mean = M.mean(0); std = M.std(0, ddof=1) if len(M) > 1 else np.zeros_like(mean)
    any_ = mean[PAIRS] > 0
    cv = std[PAIRS][any_] / mean[PAIRS][any_]
    pc = np.array([s["pearson_count"] for s in scores])
    return dict(mean=mean, std=std, n=len(M), pearson_mean=float(pc.mean()), pearson_std=float(pc.std(ddof=1)) if len(pc) > 1 else 0.0,
                cv_median=float(np.median(cv)) if cv.size else float("nan"), pairs_any=int(any_.sum()),
                pairs_always=int((M[:, PAIRS[0], PAIRS[1]] > 0).all(0).sum()))


def dti(dwi, meas, mask, *, b_max=1500.0):
    """``(md, fa)`` grids from a log-linear tensor fit on the rows with b <= ``b_max`` s/mm^2 (the b = 0 rows
    included): mean diffusivity in um^2/ms and fractional anisotropy, NaN outside ``mask``."""
    rows = meas.bvals <= b_max
    if rows.sum() < 7 or meas.b0[rows].sum() < 1 or (~meas.b0[rows]).sum() < 6:
        raise ValueError("a tensor fit needs a b = 0 row and six or more directions at b <= b_max")
    g = meas.dirs[rows]; b = meas.bvals[rows] / 1e3                    # ms/um^2, so D comes out in um^2/ms
    X = np.column_stack([np.ones(rows.sum()), -b * g[:, 0] ** 2, -b * g[:, 1] ** 2, -b * g[:, 2] ** 2,
                         -2 * b * g[:, 0] * g[:, 1], -2 * b * g[:, 0] * g[:, 2], -2 * b * g[:, 1] * g[:, 2]])
    S = np.clip(np.nan_to_num(dwi[mask][:, rows], nan=1e-6), 1e-6, None)
    beta = np.linalg.pinv(X) @ np.log(S).T                              # (7, n_vox)
    D = np.empty((beta.shape[1], 3, 3))
    D[:, 0, 0] = beta[1]; D[:, 1, 1] = beta[2]; D[:, 2, 2] = beta[3]
    D[:, 0, 1] = D[:, 1, 0] = beta[4]; D[:, 0, 2] = D[:, 2, 0] = beta[5]; D[:, 1, 2] = D[:, 2, 1] = beta[6]
    ev = np.linalg.eigvalsh(D)
    md = ev.mean(1)
    fa = np.sqrt(1.5 * ((ev - md[:, None]) ** 2).sum(1) / np.maximum((ev ** 2).sum(1), 1e-30))
    MD = np.full(mask.shape, np.nan); FA = np.full(mask.shape, np.nan)
    MD[mask] = md; FA[mask] = fa
    return MD, FA


def layer_differences(layers, meas, mask):
    """Between consecutive layers ``[(label, dwi)]``, per shell: the median and 99th percentile of |dS| over the
    voxels of ``mask``; rows ``(from, to, b, median, p99)``."""
    rows = []
    shells = np.unique(np.round(meas.bvals[~meas.b0]))
    for (la, a), (lb, b) in zip(layers[:-1], layers[1:]):
        d = np.abs(b - a)[mask]
        for sh in shells:
            cols = np.round(meas.bvals) == sh
            v = d[:, cols][np.isfinite(d[:, cols])]
            rows.append((la, lb, float(sh), float(np.median(v)) if v.size else float("nan"), float(np.quantile(v, 0.99)) if v.size else float("nan")))
    return rows


def floor_stats(res):
    """The run's split-half floor over the voxels with signal: ``median``, ``p99``, ``max`` and the voxel count."""
    f = res.floor[np.isfinite(res.dwi[..., 0])]
    return dict(median=float(np.median(f)), p99=float(np.quantile(f, 0.99)), max=float(f.max()), n=int(f.size))


def b0_snr(res):
    """The b = 0 SNR of the run's voxels with signal (``snr`` at M0 times each voxel's S0 factor): ``median``,
    ``min``, ``max``; None for a noiseless run."""
    if res.snr is None:
        return None
    f = res.s0_factor[np.isfinite(res.dwi[..., 0])] * res.snr
    return dict(median=float(np.nanmedian(f)), min=float(np.nanmin(f)), max=float(np.nanmax(f)))


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
    it: rows of ``(shell, b, delta, Delta, G_needed, G_max or None, ok)`` in SI. A stimulated-echo class needs the
    same amplitude as its PGSE twin (the same δ and diffusion time)."""
    G_max = scanner_classes()[scanner][0] if scanner else None
    rows = []
    for s in protocol.shells:
        t = dict(delta=s.delta, Delta=s.Delta) if s.free_timing else shapes[s.shape]
        delta, Delta = float(t["delta"]), float(t["Delta"])
        G = gradient_needed(s.b, delta, Delta)
        rows.append((s.shape, s.b, delta, Delta, G, G_max, G_max is None or G <= G_max))
    return rows


def sample_tractogram(tg, n, *, seed=0):
    """``n`` streamlines drawn without replacement from ``tg`` (``tg`` itself when it has no more), in their
    original order: what the page keeps and draws while the full tractogram goes to the .tck file."""
    if len(tg) <= n:
        return tg
    return tg.select(np.sort(np.random.default_rng(seed).choice(len(tg), n, replace=False)))


@dataclass
class Result:
    """One run: the protocol and its rows, the DWI the CSD saw (noisy when ``snr`` is set), the replay floor per
    voxel, the voxel's b = 0 signal over M0 (what the physics took, the b = 0 SNR is ``snr`` times it), the FOD SH
    field, the tractogram and its seeds, the connectome and its score, the stage times, the physics (None for bare
    diffusion), and ``clean``, the replay before the noise."""
    protocol: Protocol
    meas: Measurements
    dwi: np.ndarray
    floor: np.ndarray
    s0_factor: np.ndarray
    sh: np.ndarray
    tractogram: object
    seeds: np.ndarray
    matrix: np.ndarray
    score: dict
    seconds: dict = field(default_factory=dict)
    clean: Optional[np.ndarray] = None
    snr: Optional[float] = None
    physics: Optional[Physics] = None


STAGES = ("replay", "noise", "csd", "track", "score")


def run_stages(source, protocol, *, snr=None, tracking=Tracking(), noise_seed=0, physics=None):
    """The pipeline as a generator: before each of the ``STAGES`` it yields ``(stage, k, n)`` (so a page can show
    where the run is, from inside any worker), and last the :class:`Result`. ``physics`` is the tissue and scanner
    the replay is evaluated at (None: bare diffusion)."""
    t_run = time.perf_counter()

    def at(stage):
        print(f"[pipeline] {stage} at +{time.perf_counter() - t_run:.1f} s", flush=True)      # the server log shows where a run is
        return stage, STAGES.index(stage), len(STAGES)
    meas = source.validate(protocol, physics)
    yield at("replay"); dwi, floor, s0_factor, t_replay = source.replay(meas, physics)
    yield at("noise"); t0 = time.perf_counter(); noisy = add_noise(dwi, snr, s0_factor, seed=noise_seed, backend=source.backend); t_noise = time.perf_counter() - t0
    signal = np.isfinite(dwi[..., 0])
    yield at("csd"); sh, t_csd = csd(noisy, meas, signal, backend=source.backend)
    yield at("track"); t0 = time.perf_counter(); fld, seeds = tracking_inputs(sh, source, tracking.density); tg, _ = track(fld, seeds, tracking, source.backend); t_track = time.perf_counter() - t0
    yield at("score"); t0 = time.perf_counter(); M = connectome(tg, source); s = score(M, source); t_score = time.perf_counter() - t0
    yield Result(protocol, meas, noisy, floor, s0_factor, sh, tg, seeds, M, s, snr=snr, physics=physics, clean=dwi,
                 seconds=dict(replay=t_replay, noise=t_noise, csd=t_csd, track=t_track, score=t_score,
                              total=t_replay + t_noise + t_csd + t_track + t_score))


def run(source, protocol, *, snr=None, tracking=Tracking(), noise_seed=0, physics=None, progress=None):
    """The whole pipeline for one protocol: every stage's output and time. ``progress(stage, k, n)`` is called as
    each stage begins."""
    for item in run_stages(source, protocol, snr=snr, tracking=tracking, noise_seed=noise_seed, physics=physics):
        if isinstance(item, Result):
            return item
        if progress:
            progress(*item)


def repeat_tracking(res, source, tracking, keys):
    """The tractography of ``res`` (its FOD field, the seeds built once) repeated over tracker ``keys``: a generator
    of ``(key, matrix, score, seconds)``; the replay, noise and CSD are kept, only the tracker's randomness varies."""
    fld, seeds = tracking_inputs(res.sh, source, tracking.density)
    for k in keys:
        t0 = time.perf_counter()
        tg, _ = track(fld, seeds, replace(tracking, key=int(k)), source.backend)
        M = connectome(tg, source)
        yield int(k), M, score(M, source), time.perf_counter() - t0
