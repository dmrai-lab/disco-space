"""The Spaces' pipeline: an acquisition on a replay source -> the DWI of the source's grid -> noise -> the
reconstruction -> probabilistic tracking -> the connectome of the source's regions against the source's truth. Plain
functions with their timings; nothing here draws or reads a widget.

The source is what the configuration names (:mod:`space.sources`): DiSCo's replay layout or columnar pack
(:mod:`space.sources.disco`), or a brain composed from an FOD field, tissue fractions and replay packs
(:mod:`space.sources.brain`). Everything a page needs from a source -- its texts, presets, tissue panel, regions,
truth, reconstruction and GPU budget -- is the :class:`Source` interface, so one page serves both. The CSD is
dmipy-fit's, the tracker dmipy-tract's ``track``, the connectome dmipy-tract's ``connectivity`` over the source's
label image.
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
DEFAULT_CONFIG = os.path.join(HERE, "config.toml")
DATA_DIR = os.path.join(ROOT, "data")
SH_ORDER = 8
B0_MAX = 50.0                                        # s/mm^2: a row below it is a b = 0 measurement, everywhere in the pipeline
BAND_TOL = 0.005                                     # the band tolerance (x the pack's floor) of the layout and of a full replay
BACKENDS = ("jax", "torch")
MODES = ("demo", "full")
RECONSTRUCTIONS = ("msmt", "tournier07")
NO_KNOB = "none: run A only"                        # the knob that makes no B


def config_path():
    """The configuration this process serves: ``DISCO_CONFIG`` in the environment (a path, or a name in
    ``space/`` such as ``brain.toml``), else ``space/config.toml`` (the DiSCo Space)."""
    p = os.environ.get("DISCO_CONFIG") or DEFAULT_CONFIG
    return p if os.path.isabs(p) or os.path.exists(p) else os.path.join(HERE, p)


def config(path=None):
    """The configuration at ``path``, else at :func:`config_path`."""
    with open(path or config_path(), "rb") as f:
        return tomllib.load(f)


def reconstruction(cfg):
    """The reconstruction a run fits: ``msmt`` (three-tissue responses estimated from the data and multi-shell
    multi-tissue CSD) or ``tournier07`` (the single-fibre response and single-tissue CSD): ``DISCO_RECONSTRUCTION``
    in the environment, else ``[reconstruction] method``."""
    r = os.environ.get("DISCO_RECONSTRUCTION") or cfg["reconstruction"]["method"]
    if r not in RECONSTRUCTIONS:
        raise ValueError(f"reconstruction must be one of {RECONSTRUCTIONS}, got {r!r}")
    return r


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


def preset_protocol(cfg, name):
    """A ``[presets]`` entry of the config as a Protocol."""
    p = cfg["presets"][name]
    return Protocol(tuple(Shell(s["shape"], float(s["b"]), int(s["n_dirs"])) for s in p["shells"]), n_b0=int(p["n_b0"]), name=name)


def retime(protocol, shape):
    """``protocol`` with every shell on the timing class ``shape`` (b-values, directions and counts kept): the A/B
    knob that swaps a PGSE class for its stimulated-echo twin, or one δ / Δ for another."""
    return Protocol(tuple(Shell(shape, s.b, s.n_dirs) for s in protocol.shells), n_b0=protocol.n_b0,
                    directions=protocol.directions, bvals=protocol.bvals, name=f"{protocol.name} on {shape}")


def write_volumes(res, out_dir, *, prefix, affine):
    """The result's DWI as NIfTI (float32 on ``affine``, NaN as 0) with its bvals (s/mm^2) and bvecs (3 x N), and
    the FOD SH field as NIfTI (tournier07 basis, order 8); returns the four paths."""
    import nibabel as nib
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    dwi = os.path.join(out_dir, f"{prefix}_dwi.nii.gz")
    nib.save(nib.Nifti1Image(np.nan_to_num(res.dwi).astype(np.float32), np.asarray(affine, np.float64)), dwi); paths["dwi"] = dwi
    bvals = os.path.join(out_dir, f"{prefix}.bvals"); np.savetxt(bvals, res.meas.bvals[None, :], fmt="%.1f"); paths["bvals"] = bvals
    bvecs = os.path.join(out_dir, f"{prefix}.bvecs"); np.savetxt(bvecs, res.meas.dirs.T, fmt="%.8f"); paths["bvecs"] = bvecs
    fod = os.path.join(out_dir, f"{prefix}_fod_sh.nii.gz")
    nib.save(nib.Nifti1Image(np.nan_to_num(res.sh).astype(np.float32), np.asarray(affine, np.float64)), fod); paths["fod"] = fod
    return paths


# ---- the physics: tissue and scanner --------------------------------------------------------------------------------
CATALOGUE_FIELDS = (1.5, 3.0, 7.0)                       # where dmipy-sim's white-matter catalogue has cited relaxation
CATALOGUE_POOLS = ("intra", "extra", "myelin")           # the pools the catalogue has T2 and T1 for
B0_ALONG_Z = (0.0, 0.0, 1.0)
B0_TRANSVERSE = (1.0, 0.0, 0.0)


def catalogue(field_T, pools=CATALOGUE_POOLS):
    """dmipy-sim's canonical white matter at the catalogue field nearest ``field_T`` (in log distance): ``T2`` and
    ``T1`` per pool of ``pools`` (s), ``rho2`` (m/s), the sheath's ``chi_iso`` and ``chi_aniso`` (SI), and
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
                rho2=float(w["rho2"]), chi_iso=float(w["chi_iso_myelin"]), chi_aniso=float(w["delta_chi_a"]))


def b0_direction(theta_deg, phi_deg):
    """The unit field direction at polar angle ``theta`` from z and azimuth ``phi`` from x (degrees)."""
    t, f = np.radians(float(theta_deg)), np.radians(float(phi_deg))
    return (float(np.sin(t) * np.cos(f)), float(np.sin(t) * np.sin(f)), float(np.cos(t)))


@dataclass(frozen=True)
class Physics:
    """The tissue and the scanner a replay is evaluated at: the field (T) and its direction in the substrate frame,
    T2 and T1 per seeded pool (s), the walls' surface relaxivity ``rho2`` (m/s), the sheath's ``chi_iso`` and
    ``chi_aniso`` (SI, the field source); ``relaxation`` / ``contact`` / ``field`` switch the three tiers, so a tier
    is a knob the page can turn off one at a time. :meth:`tissue` is the :class:`dmipy_sim.spec.tissue.Tissue` for
    the replay (None when every tier is off: bare diffusion). ``scanner`` is the machine (a catalogue key; None: the
    ideal scanner), whose field the physics is at, and ``offset_m`` where the phantom's centre sits from isocentre
    (metres, the scanner's frame): together they say what gradient every voxel is delivered (docs/scanner.md)."""
    field_T: float
    T2: dict
    T1: dict
    rho2: float
    chi_iso: float
    chi_aniso: float
    b0_direction: tuple = B0_ALONG_Z
    relaxation: bool = True
    contact: bool = True
    field: bool = True
    scanner: Optional[str] = None
    offset_m: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.scanner is not None:
            L = limits(self.scanner)
            if abs(float(L.field_T) - float(self.field_T)) > 1e-9:
                raise ValueError(f"{self.scanner} is a {L.field_T:g} T magnet; the physics says {self.field_T:g} T")
        if np.shape(self.offset_m) != (3,):
            raise ValueError("offset_m is the phantom centre's displacement from isocentre, three lengths in metres")
        if not (0 < self.field_T < 30):
            raise ValueError(f"the field is in tesla, got {self.field_T}")
        if not self.T2 or set(self.T1) != set(self.T2):
            raise ValueError(f"T2 and T1 are seconds over the same pools, got {self.T2} and {self.T1}")
        for what, m in (("T2", self.T2), ("T1", self.T1)):
            if any(not (0 < float(v) < 100) for v in m.values()):
                raise ValueError(f"{what} is seconds per pool, got {m}")
        if self.rho2 < 0:
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
                      rho2=self.rho2 if self.contact else None, chi_iso=self.chi_iso if self.field else None,
                      chi_aniso=self.chi_aniso if self.field else 0.0)

    def label(self):
        where = "" if self.scanner is None else f" on the {self.scanner} at ({', '.join(f'{x * 100:.1f}' for x in self.offset_m)}) cm"
        if self.bare:
            return "bare diffusion" + where
        tiers = [n for n, on in (("relaxation", self.relaxation), ("contact", self.contact), ("field", self.field)) if on]
        u = self.b0_direction
        return f"{self.field_T:g} T along ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f}), tiers {'+'.join(tiers)}" + where


def tissue_and_scanner(physics, unseeded=()):
    """``(tissue, scanner)`` for a replay at ``physics``: ``(None, None)`` for bare diffusion; the scanner's field
    goes with the field tier only (relaxation at a field is already in the tissue's T2 and T1)."""
    tissue = physics.tissue(unseeded) if physics else None
    return tissue, (physics.field_T if tissue is not None and physics.field else None)


# ---- the sources: what replays, what is tracked, what is scored ------------------------------------------------------

@dataclass(frozen=True)
class Control:
    """One widget of a source's tissue-and-scanner panel: ``kind`` is an input (``checkbox``, ``number``, ``slider``,
    ``dropdown``) or a piece of the panel's machinery (``field_preset``: the field presets that fill the catalogue's
    numbers, ``catalogue_note`` and ``reset``: the note saying where the numbers came from and the button that
    restores them, ``markdown``: text). ``value`` is the default; ``interactive`` False shows it fixed."""
    name: str
    kind: str
    label: str = ""
    value: object = None
    choices: tuple = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    visible: bool = True
    interactive: bool = True


@dataclass(frozen=True)
class Panel:
    """A source's tissue-and-scanner panel: ``rows`` of :class:`Control` in the order drawn, ``fields`` the inputs'
    names in the order the run signature carries them, ``catalogue`` the names the catalogue fills in (in the order
    :meth:`Source.catalogue_numbers` returns them, a note last), ``field_presets`` the fields (T) offered."""
    rows: tuple
    fields: tuple
    catalogue: tuple
    field_presets: tuple

    def controls(self):
        return {c.name: c for row in self.rows for c in row}


@dataclass(frozen=True)
class Regions:
    """The regions a connectome is counted between: ``labels`` an integer image (0 = no region, 1..n the regions),
    ``affine`` its voxel-to-tracking-frame map in millimetres, ``names`` one per region, ``groups`` the coarser view
    as ``((name, (label, ...)), ...)`` (empty when the source has none)."""
    labels: np.ndarray
    affine: np.ndarray
    names: tuple
    groups: tuple = ()

    @property
    def n(self):
        return len(self.names)

    @property
    def pairs(self):
        """The region pairs, the upper triangle without the diagonal."""
        return np.triu_indices(self.n, 1)

    def matrix(self, tg):
        """The symmetrised ``n x n`` streamline-count matrix of ``tg`` between the regions (no self-connections):
        dmipy-tract's ``connectivity`` over the label image, row 0 (no region) dropped."""
        from dmipy_tract import connectivity
        counts, _ = connectivity(tg, self.labels, self.affine)
        M = np.zeros((self.n + 1, self.n + 1))
        k = min(counts.shape[0], self.n + 1)
        M[:k, :k] = counts[:k, :k]
        M = M[1:, 1:]
        M = M + M.T
        np.fill_diagonal(M, 0)
        return M

    def grouped(self, M):
        """``M`` summed over the groups: ``(G, G)``, the off-diagonal entries the counts between two groups, the
        diagonal the count within a group (each pair once)."""
        A = np.zeros((self.n, len(self.groups)))
        for g, (_, ids) in enumerate(self.groups):
            A[np.asarray(ids, int) - 1, g] = 1.0
        G = A.T @ np.asarray(M, np.float64) @ A
        G[np.diag_indices_from(G)] /= 2.0
        return G


class Source:
    """A replay source: what the page shows and asks for (class-level, from the configuration alone, so the page is
    built before any data loads) and what a run does with the data (the instance).

    The page's side: :meth:`describe` (the title and every text that names the source), :meth:`presets` and
    :meth:`protocol` (the acquisitions offered), :meth:`panel` (the tissue-and-scanner inputs),
    :meth:`catalogue_numbers`, :meth:`physics_from`, :meth:`knobs` and :meth:`apply_knob` (B, one knob away from A),
    :meth:`tracking_controls` and :meth:`estimated_seconds` (the GPU seconds a run reserves on a shared pool).

    The run's side: :attr:`regions` and :attr:`affine`, :meth:`validate`, :meth:`prepare` (the share of a replay
    that needs no device, kept per process: :meth:`response_key`, :meth:`cached`, :meth:`keep`,
    :meth:`responses`), :meth:`replay`, :meth:`reconstruct`, :meth:`tracking_inputs`,
    :meth:`reference` and :meth:`score` (the truth the connectome is scored against), :meth:`compare` (A against
    B), :meth:`ladder`, :meth:`ingredients`, :meth:`accuracy`."""
    mode = ""
    STAGES = ("replay", "noise", "csd", "track", "score")
    TIERS = ()                                       # the physics tiers a ladder switches on one at a time
    unseeded = ()                                    # the substrate's pools no walker was seeded in

    def __init__(self, cfg):
        self.cfg = cfg
        self.shapes = cfg["shapes"]
        self.backend = backend(cfg)

    # ---- the page's side: from the configuration alone ----
    @classmethod
    def describe(cls, cfg):
        """The texts that name the source: ``title``, ``heading`` (the page's first paragraph), ``acquisition``
        (under the shell rows), ``tissue`` (under the panel), ``explorer`` (the explorer's paragraph), ``results_tab``,
        ``truth_tab`` (the tab of the input and its truth), ``stages`` (the progress text per stage), ``fixed`` (what a
        visitor cannot change, one line each), ``accuracy`` (the accuracy accordion's paragraph), ``ingredients``
        (the three ingredient images' labels), ``truth`` (what the connectome is scored against, and how),
        ``files`` (the download files' prefix)."""
        raise NotImplementedError

    @classmethod
    def presets(cls, cfg):
        """The acquisitions the page offers besides custom shells, the default first."""
        raise NotImplementedError

    @classmethod
    def shapes_of(cls, cfg):
        """The timing classes a shell can play: ``{name: {label, delta, Delta, TE[, kind]}}``."""
        return cfg["shapes"]

    @classmethod
    def protocol(cls, cfg, name):
        """The preset ``name`` as a :class:`Protocol`."""
        raise NotImplementedError

    @classmethod
    def panel(cls, cfg):
        """The tissue-and-scanner :class:`Panel`."""
        raise NotImplementedError

    @classmethod
    def catalogue_numbers(cls, cfg, field_T):
        """The panel's :attr:`Panel.catalogue` numbers at ``field_T`` in the page's units, a note last."""
        raise NotImplementedError

    #: Whether the source plays its pulses at the page's scanner gradient class's slew (:meth:`physics_from` takes
    #: ``gradient``); a source that does not plays ideal pulses and the class only checks what it can play.
    plays_slew = False

    @classmethod
    def physics_from(cls, cfg, values, gradient=None):
        """The panel (a dict by :attr:`Panel.fields`, page units) as the source's physics, or None when it is off;
        ``gradient`` the page's scanner gradient class (a catalogue name) for a source with :attr:`plays_slew`, None
        for ideal pulses."""
        raise NotImplementedError

    @classmethod
    def knobs(cls, cfg):
        """The one-knob changes B can make to A, ``{label: (kind, value)}`` (None for no B)."""
        raise NotImplementedError

    @classmethod
    def apply_knob(cls, cfg, change, protocol, snr_on, snr, values):
        """B's settings: A's with ``change`` applied; ``(protocol, snr_on, snr, values)``."""
        raise NotImplementedError

    @classmethod
    def tracking_controls(cls, cfg):
        """The tracker's inputs: ``{name: (minimum, maximum, value, step, label)}`` for ``density``,
        ``max_angle`` and ``step``."""
        raise NotImplementedError

    @classmethod
    def tracking(cls, cfg, *, density, max_angle, step, key):
        """The tracker's :class:`Tracking` from the page's inputs."""
        raise NotImplementedError

    @classmethod
    def estimated_seconds(cls, cfg, protocol, *, density, knob, n_keys, ladder, responses=(), scanner=None):
        """The GPU seconds a run of ``protocol`` reserves on a shared pool, from the measured cost model;
        ``responses`` the ``(state, n_meas, saves)`` of each :meth:`prepare` the call computes (:meth:`responses`),
        ``scanner`` the machine (a catalogue key; None: the ideal scanner)."""
        raise NotImplementedError

    # ---- the run's side ----
    @classmethod
    def load(cls, cfg, *, local=None):
        """The source of ``cfg`` with its data loaded (``local``: a local copy instead of the Hub's)."""
        return cls(cfg)

    def validate(self, protocol, physics):
        """Refuses, before any work, a run this source cannot do; returns its measurements."""
        return measurements(protocol, self.shapes)

    def response_key(self, meas, physics=None):
        """The key of :meth:`prepare`'s result in this process's cache; None for a source with nothing to prepare."""
        return None

    def cached(self, meas, physics=None):
        """:meth:`prepare`'s result for ``meas`` at ``physics`` when this process has it, else None."""
        return None

    def keep(self, prepared):
        """``{response_key: result}`` of :meth:`prepare` computed in another process kept in this one's cache."""
        return

    def responses(self, entries):
        """``(state, n_meas, saves)`` for each :meth:`prepare` of a run's ``entries`` ``[(meas, physics)]`` that a
        process forked from this one would compute, ``state`` a ``[budget]`` key, ``saves`` the pack saves it spans;
        empty for a source with nothing to prepare."""
        return []

    def prepare(self, meas, physics=None):
        """The share of a replay of ``meas`` at ``physics`` that needs no device, computed and kept in this process's
        cache (None for a source that has none); :meth:`replay` takes it as ``prepared``. A run looks it up on the page
        (:meth:`cached`) and computes what is missing inside the GPU call."""
        return None

    def replay(self, meas, physics=None, prepared=None):
        """``(dwi, floor, s0_factor, seconds)``: the S0-normalised signal of every voxel per measurement (NaN
        outside the source's voxels), the split-half floor per voxel (zero where the source has none), the voxel's
        raw b = 0 signal as a fraction of M0 (what relaxation, contact and the field took, 1 when the source cannot
        tell), and the time; at ``physics`` else bare diffusion; ``prepared`` is :meth:`prepare`'s."""
        raise NotImplementedError

    def reconstruct(self, dwi, s0_factor, meas, mask):
        """``(sh, seconds, extras)``: the FOD field from the DWI, the time of each step, and whatever else the
        reconstruction estimated (``fractions`` on the multi-tissue path): the configuration's
        :func:`reconstruction`, the single-fibre response and dmipy-fit's batched Tournier 2007 CSD (:func:`csd`, one
        step ``csd``) or three-tissue responses and multi-tissue CSD on the signal in M0 units (:func:`csd_msmt`)."""
        if reconstruction(self.cfg) == "tournier07":
            sh, t, _ = csd(dwi, meas, mask, backend=self.backend)
            return sh, {"csd": t}, {}
        sh, fractions, _, (t_resp, t_fit) = csd_msmt(dwi * s0_factor[..., None], meas, mask, backend=self.backend)
        return sh, {"responses (dhollander16)": t_resp, f"MT-CSD (csd_msmt_{self.backend})": t_fit}, dict(fractions=fractions)

    def tracking_inputs(self, sh, density):
        """``(field, seeds)``: the FOD field on the tracking domain and the seeds at ``density``."""
        raise NotImplementedError

    def reference(self, tracking, seeds):
        """``(reference, seconds)``: what the connectome is scored against at ``tracking`` from ``seeds``."""
        raise NotImplementedError

    def score(self, M, reference):
        """The connectome ``M`` against ``reference``: a dict of numbers."""
        raise NotImplementedError

    def compare(self, a, b):
        """Run A against run B (two :class:`Result`): a dict of numbers."""
        raise NotImplementedError

    def plan(self, meas, physics=None):
        """What a replay reads before any byte moves (None when the source has nothing to plan)."""
        return None

    def warm(self):
        """Whatever the first run would otherwise pay (nothing for a source that reads per run)."""
        return

    def warm_in_background(self):
        """:meth:`warm` while the page serves, for a source whose warm-up is long enough to matter; this one's is
        :meth:`warm` itself, done before the page serves. Returns the handle the caller may join, or None."""
        self.warm()
        return None

    def release(self):
        """Whatever a run kept on the device that must not survive it (nothing for a source that reads per run)."""
        return

    def ingredients(self, meas, physics=None, prepared=None):
        """The per-voxel maps of what each tier multiplies into the sum (None for a source without them)."""
        return None

    def ladder_steps(self, physics):
        """The rungs below ``physics``: ``[(label, physics)]`` from bare diffusion (None) with the source's
        :attr:`TIERS` switched on one at a time, up to (not including) ``physics`` itself; empty for bare physics."""
        if physics is None or physics.bare:
            return []
        steps = [("bare diffusion", None)]
        on = []
        wanted = [q for q in self.TIERS if getattr(physics, q)]
        for tier in wanted[:-1]:
            on.append(tier)
            steps.append(("+ " + " + ".join(on), replace(physics, **{q: q in on for q in self.TIERS})))
        return steps

    def ladder(self, meas, physics, prepared=None):
        """The noise-free replays of :meth:`ladder_steps`, ``[(label, dwi)]``; ``prepared`` is one :meth:`prepare`
        per rung, in order (None for a source that has none)."""
        steps = self.ladder_steps(physics)
        prepared = prepared if prepared is not None else [None] * len(steps)
        return [(label, self.replay(meas, ph, pr)[0]) for (label, ph), pr in zip(steps, prepared)]

    # ---- what the page draws from a run (None where the source has no such view) ----
    def score_key(self):
        """The score's headline number, by its key in :meth:`score`'s dict."""
        raise NotImplementedError

    def score_text(self, tag, res):
        """The headline of run ``tag`` (Markdown)."""
        raise NotImplementedError

    def compare_text(self, c, knob):
        """The headline line of A against B (:meth:`compare`'s dict)."""
        raise NotImplementedError

    def matrices(self, M, score, reference):
        """The connectome beside what it is scored against (an image)."""
        raise NotImplementedError

    def truth_views(self):
        """The truth tab's two views (a 3-D figure, an image), drawn once per process."""
        raise NotImplementedError

    @staticmethod
    def ingredient_layers(ingredients):
        """The explorer's three ingredient images as ``[(title, map, style)]`` (``style`` the keywords of
        :func:`space.viewers.map_slice`, ``map`` None for an image with nothing to show)."""
        return [None, None, None]

    def lobar(self, M, score, reference):
        """The connectome over the regions' groups beside the reference's (an image), None without groups."""
        return None

    def truth_view(self, dwi, meas, z, m):
        """A DWI slice with the input's FOD peaks over it, None for a source whose input is not an FOD."""
        return None

    def fractions_view(self, extras, z):
        """The recovered tissue fractions against the input's at slice ``z``, None for a source without them."""
        return None

    def roundtrip_rows(self, res):
        """Rows ``(what, value)`` of the reconstruction against the input; empty for a source without one."""
        return []

    def response_view(self, prepared, extras):
        """The estimated responses against the true ones (an image), None for a source without them."""
        return None

    def extra_files(self, res, out_dir, stem):
        """Files beside the tractograms and volumes (paths)."""
        return []

    def accuracy(self, res=None):
        """Rows ``(what, value)`` saying what this source's replay is an approximation of, and ``res``'s own floor."""
        rows = []
        if res is not None:
            if np.any(res.floor > 0):
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
    """``(sh, seconds, response)``: the FOD field ``(X, Y, Z, 45)`` in the tournier07 basis, dmipy-fit's batched
    Tournier 2007 solver on ``backend`` with the response ``white_matter_response_tournier07`` estimates from the
    masked voxels (``response`` is its ``(S0, model)``), fitted on ``mask`` (the voxels with signal)."""
    from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
    from dmipy_fit.tissue_response.white_matter_response import white_matter_response_tournier07
    t0 = time.perf_counter()
    data = np.nan_to_num(dwi, nan=0.0)
    sch = scheme(meas)
    S0, response, _ = white_matter_response_tournier07(sch, data[mask])
    mc = MultiCompartmentSphericalHarmonicsModel(models=[response], sh_order=SH_ORDER)
    fitted = mc.fit(sch, data, mask=mask, solver=f"csd_tournier07_{backend}", verbose=False)
    return np.asarray(fitted.fitted_parameters["sh_coeff"], np.float64), time.perf_counter() - t0, (S0, response)


MSMT_ISSUE = "dmrai-lab/dmipy-fit#39"


def csd_msmt(data, meas, mask, backend="torch"):
    """``(sh, fractions, responses, seconds)``: three-tissue responses estimated from the data (Dhollander 2016,
    ``three_tissue_response_dhollander16`` with ``mask=``; the white-matter response by Tournier 2007's FA selection,
    the single-tissue path's, since Tournier 2013's iteration does not converge on a replayed brain in its five
    iterations and costs 5 s where this costs 0.2 s) and multi-shell multi-tissue CSD (Jeurissen 2014,
    ``solver='csd_msmt_<backend>'``), dmipy-fit's batched solvers of dmipy-fit#39. ``data`` is the signal in M0 units
    (not S0-normalised: the tissues' b = 0 signals are what separates them); ``sh`` the WM FOD ``(X, Y, Z, 45)``,
    ``fractions`` the fitted WM / GM / CSF volume fractions ``(X, Y, Z, 3)``, ``responses`` ``(S0s, models)`` as
    estimated, ``seconds`` ``(responses, fit)``. Refuses by name, before any work, a dmipy-fit without them."""
    import inspect
    from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
    from dmipy_fit.tissue_response.three_tissue_response import three_tissue_response_dhollander16
    params = inspect.signature(three_tissue_response_dhollander16).parameters
    if "mask" not in params or "backend" not in params:
        raise RuntimeError(f"this dmipy-fit has no batched dhollander16 (three_tissue_response_dhollander16(scheme, data, mask=, backend=)): "
                           f"the multi-tissue reconstruction is {MSMT_ISSUE}; until it merges, DISCO_RECONSTRUCTION=tournier07 "
                           f"runs the single-tissue Tournier 2007 CSD instead")
    t0 = time.perf_counter()
    data = np.nan_to_num(np.asarray(data, np.float64), nan=0.0)
    sch = scheme(meas)
    S0s, models, _ = three_tissue_response_dhollander16(sch, data, mask=mask, backend=backend, wm_algorithm="tournier07")
    t_resp = time.perf_counter() - t0; t0 = time.perf_counter()
    mc = MultiCompartmentSphericalHarmonicsModel(models=list(models), S0_tissue_responses=list(S0s), sh_order=SH_ORDER)
    try:
        fitted = mc.fit(sch, data, mask=mask, solver=f"csd_msmt_{backend}", verbose=False)
    except ValueError as e:
        if "Unknown solver" in str(e):
            raise RuntimeError(f"this dmipy-fit has no solver csd_msmt_{backend}: the multi-tissue CSD is {MSMT_ISSUE}") from e
        raise
    fp = fitted.fitted_parameters
    fractions = np.stack([np.asarray(fp[name], np.float64) for name in mc.partial_volume_names], -1)
    return np.asarray(fp["sh_coeff"], np.float64), fractions, (list(S0s), list(models)), (t_resp, time.perf_counter() - t0)


@dataclass(frozen=True)
class Tracking:
    """The tracker's settings: dipy's defaults for the DiSCo score."""
    density: int = 4
    step_mm: float = 0.5
    max_angle: float = 30.0
    max_steps: int = 500
    relative_threshold: float = 0.1
    key: int = 0


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


def fod_peaks(sh, n_max=3, *, relative=0.1, separation_deg=25.0, n_dirs=362, chunk=20000):
    """The FOD's peaks per voxel of ``sh (..., n_coef)``: ``(dirs (..., n_max, 3), amps (..., n_max), count (...))``,
    the local maxima of the FOD on the tracker's hemisphere (antipodes identified, a direction's neighbours those
    within 12 degrees) above ``relative`` of the voxel's largest value, strongest first, a weaker one within
    ``separation_deg`` of a stronger one dropped; zero rows beyond ``count``. The first is :func:`peaks`'s."""
    from dmipy_tract import hemisphere, sh_matrix
    V = hemisphere(n_dirs)
    B = sh_matrix(SH_ORDER, V)
    c = np.abs(V @ V.T)
    nbr = [np.flatnonzero((c[i] > np.cos(np.radians(12.0))) & (np.arange(n_dirs) != i)) for i in range(n_dirs)]
    width = max(len(x) for x in nbr)
    table = np.array([np.pad(x, (0, width - len(x)), constant_values=i) for i, x in enumerate(nbr)])
    coef = np.nan_to_num(np.asarray(sh, np.float64)).reshape(-1, np.shape(sh)[-1])
    n = coef.shape[0]
    dirs = np.zeros((n, n_max, 3)); amps = np.zeros((n, n_max)); count = np.zeros(n, np.int64)
    cos_sep = np.cos(np.radians(separation_deg))
    for s0 in range(0, n, chunk):
        A = coef[s0:s0 + chunk] @ B.T                                            # (m, n_dirs)
        top = A.max(1, keepdims=True)
        is_max = (A >= A[:, table].max(2)) & (A > relative * top) & (top > 0)
        cand = np.where(is_max, A, -np.inf)
        order = np.argsort(-cand, axis=1)[:, :n_max + 3]                         # a few spare candidates for the separation rule
        kept = np.zeros((len(A), n_max), np.int64); nk = np.zeros(len(A), np.int64)
        for j in range(order.shape[1]):
            idx = order[:, j]
            ok = np.isfinite(cand[np.arange(len(A)), idx]) & (nk < n_max)
            for q in range(n_max):                                               # not within separation of a kept one
                prev = kept[:, q]
                close = (q < nk) & (np.abs(np.einsum("ij,ij->i", V[idx], V[prev])) > cos_sep)
                ok &= ~close
            kept[ok, nk[ok]] = idx[ok]; nk[ok] += 1
        rows = np.arange(len(A))[:, None]
        valid = np.arange(n_max)[None, :] < nk[:, None]
        dirs[s0:s0 + chunk] = np.where(valid[..., None], V[kept], 0.0)
        amps[s0:s0 + chunk] = np.where(valid, A[rows, kept], 0.0)
        count[s0:s0 + chunk] = nk
    lead = np.shape(sh)[:-1]
    return dirs.reshape(lead + (n_max, 3)), amps.reshape(lead + (n_max,)), count.reshape(lead)


def pearson(a, b):
    """The Pearson correlation of two vectors, NaN when either is constant."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def connected_pairs(M):
    """How many region pairs (the upper triangle of ``M`` without the diagonal) ``M`` connects."""
    return int((np.asarray(M)[np.triu_indices(len(M), 1)] > 0).sum())


def pair_spread(matrices, scores, *, key):
    """Over repeated runs: the streamline-count mean and standard deviation per region pair, the mean and standard
    deviation of the score ``key``, and the median coefficient of variation over the pairs that any run connected."""
    M = np.stack(matrices).astype(np.float64)
    pairs = np.triu_indices(M.shape[1], 1)
    mean = M.mean(0); std = M.std(0, ddof=1) if len(M) > 1 else np.zeros_like(mean)
    any_ = mean[pairs] > 0
    cv = std[pairs][any_] / mean[pairs][any_]
    pc = np.array([s[key] for s in scores])
    return dict(mean=mean, std=std, n=len(M), key=key, pearson_mean=float(pc.mean()), pearson_std=float(pc.std(ddof=1)) if len(pc) > 1 else 0.0,
                cv_median=float(np.median(cv)) if cv.size else float("nan"), pairs_any=int(any_.sum()),
                pairs_always=int((M[:, pairs[0], pairs[1]] > 0).all(0).sum()))


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


IDEAL = "ideal: no scanner terms, any gradient amplitude"


def machines(cfg):
    """The configuration's scanner menu after :data:`IDEAL`: ``{label: catalogue key}`` (``[scanners] machines``), each a
    machine of dmipy-sim's catalogue whose every catalogued term the page applies (docs/scanner.md)."""
    return {m["label"]: m["key"] for m in (cfg.get("scanners") or {}).get("machines", [])}


def machine(cfg, label):
    """The catalogue key of the menu's ``label``, None for :data:`IDEAL`; an unknown label is refused by name."""
    if label in (None, IDEAL):
        return None
    menu = machines(cfg)
    if label not in menu:
        raise ValueError(f"no scanner {label!r}; the menu is {[IDEAL] + list(menu)}")
    return menu[label]


def limits(key):
    """The :class:`dmipy_sim.acquisition.scanners.ScannerLimits` of a catalogue key."""
    from dmipy_sim.acquisition.scanners import ScannerLimits
    return ScannerLimits.of(key)


def playable(protocol, shapes, scanner=None):
    """Per shell, the gradient it needs and whether ``scanner`` (a catalogue key, or None for no limit) can play it:
    rows of ``(shell, b, delta, Delta, G_needed, G_max or None, ok)`` in SI. A stimulated-echo class needs the same
    amplitude as its PGSE twin (the same δ and diffusion time)."""
    G_max = float(limits(scanner).G_max) if scanner else None
    rows = []
    for s in protocol.shells:
        t = dict(delta=s.delta, Delta=s.Delta) if s.free_timing else shapes[s.shape]
        delta, Delta = float(t["delta"]), float(t["Delta"])
        G = gradient_needed(s.b, delta, Delta)
        rows.append((s.shape, s.b, delta, Delta, G, G_max, G_max is None or G <= G_max))
    return rows


#: how the page names a catalogue leaf's confidence
CONFIDENCE = {"cited": "measured", "widely-quoted": "measured", "derived": "derived from a measurement",
              "inferred": "inferred from the class"}
#: the scanner terms and the catalogue leaves each is read from (section, leaf), in the page's order
TERMS = (("field strength and direction", (("frame", "b0_axis"),)),
         ("field law: its gradient g0 (and its value, a phase per voxel)", (("homogeneity", "b0_harmonic_Z2"), ("homogeneity", "b0_harmonic_Z2X"))),
         ("gradient nonlinearity L", (("gradient_nonlinearity", "d_scale_y_dx"), ("gradient_nonlinearity", "b_error_quadratic_axial"))),
         ("Maxwell (concomitant) gradient", ()),
         ("transmit scale B1", (("rf", "b1_axial_falloff"), ("rf", "b1_calibration_offset"))))


def scanner_terms(key):
    """The rows of a machine's term table: ``(term, applied, where the number comes from)``. ``applied`` is whether
    the catalogue carries the term (the page applies every one it carries); the source is the catalogue leaves'
    confidence as the page names it (:data:`CONFIDENCE`) with their citation keys, or why the term is absent."""
    from dmipy_sim.acquisition import scanner_constants as scc
    L = limits(key)
    _key, entry, _kind = scc.resolve(key)
    rows = []
    have = {"field strength and direction": L.field_T is not None,
            "field law: its gradient g0 (and its value, a phase per voxel)": L.has_field_law,
            "gradient nonlinearity L": L.has_gradient_nonlinearity,
            "Maxwell (concomitant) gradient": L.field_T is not None,
            "transmit scale B1": L.has_transmit_profile}
    for term, leaves in TERMS:
        got = []
        for sec, leaf in leaves:
            rec = (entry.get(sec) or {}).get(leaf)
            if isinstance(rec, dict) and rec.get("value") not in (None, "None") and rec.get("confidence") in CONFIDENCE:
                got.append(f"{CONFIDENCE[rec['confidence']]} ({rec.get('source_key')})")
        if term == "Maxwell (concomitant) gradient":
            src = f"exact from the field strength ({L.field_T:g} T) and the coils' linear field: Maxwell's equations, no fit"
        elif term == "field strength and direction":
            u = L.b0_axis or (0.0, 0.0, 1.0)
            src = f"{L.field_T:g} T along ({u[0]:g}, {u[1]:g}, {u[2]:g}) (R, A, S); " + "; ".join(dict.fromkeys(got))
        elif have[term]:
            src = "; ".join(dict.fromkeys(got))
        elif term == "transmit scale B1" and L.b1_brain_range:
            src = f"absent: the catalogue holds a range over a brain ({L.b1_brain_range[0]:g}-{L.b1_brain_range[1]:g}), no map"
        else:
            src = "absent: the catalogue publishes no such law for this machine"
        rows.append((term, bool(have[term]), src))
    return rows


def scanner_text(cfg, label):
    """The menu entry's term table as Markdown (the page shows it under the menu)."""
    key = machine(cfg, label)
    if key is None:
        return "**ideal:** the commanded gradient everywhere, the field preset and its direction as set; no transmit scale."
    lines = [f"**{label}** (`{key}`): every term its catalogue entry carries, at the phantom's place in the bore.",
             "", "| term | applied | from |", "|---|---|---|"]
    for term, on, src in scanner_terms(key):
        lines.append(f"| {term} | {'yes' if on else 'no'} | {src} |")
    return "\n".join(lines)


def sample_tractogram(tg, n, *, seed=0):
    """``n`` streamlines drawn without replacement from ``tg`` (``tg`` itself when it has no more), in their
    original order: what the page keeps and draws while the full tractogram goes to the .tck file."""
    if len(tg) <= n:
        return tg
    return tg.select(np.sort(np.random.default_rng(seed).choice(len(tg), n, replace=False)))


@dataclass
class Result:
    """One run: the protocol and its rows, the DWI the reconstruction saw (noisy when ``snr`` is set), the replay floor
    per voxel, the voxel's b = 0 signal over M0 (what the physics took, the b = 0 SNR is ``snr`` times it), the FOD SH
    field, the tractogram and its seeds, the connectome and its score, the stage times, the physics (None for bare
    diffusion), ``clean`` (the replay before the noise), ``reference`` (what the connectome was scored against) and
    ``extras`` (what the source's reconstruction and scoring estimated beside the FOD, by name)."""
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
    physics: Optional[object] = None
    reference: Optional[object] = None
    extras: dict = field(default_factory=dict)


def run_stages(source, protocol, *, snr=None, tracking=Tracking(), noise_seed=0, physics=None, prepared=None):
    """The pipeline as a generator: before each of the source's ``STAGES`` it yields ``(stage, k, n)`` (so a page
    can show where the run is, from inside any worker), and last the :class:`Result`. ``physics`` is the tissue and
    scanner the replay is evaluated at (None: bare diffusion); ``prepared`` is the source's :meth:`Source.prepare`
    for it, done on the host before."""
    t_run = time.perf_counter()
    stages = source.STAGES

    def at(stage):
        print(f"[pipeline] {stage} at +{time.perf_counter() - t_run:.1f} s", flush=True)      # the server log shows where a run is
        return stage, stages.index(stage), len(stages)
    meas = source.validate(protocol, physics)
    yield at("replay"); dwi, floor, s0_factor, t_replay = source.replay(meas, physics, prepared)
    yield at("noise"); t0 = time.perf_counter(); noisy = add_noise(dwi, snr, s0_factor, seed=noise_seed, backend=source.backend); t_noise = time.perf_counter() - t0
    signal = np.isfinite(dwi[..., 0])
    yield at("csd"); sh, t_csd, extras = source.reconstruct(noisy, s0_factor, meas, signal)
    yield at("track"); t0 = time.perf_counter(); fld, seeds = source.tracking_inputs(sh, tracking.density); tg, _ = track(fld, seeds, tracking, source.backend); t_track = time.perf_counter() - t0
    seconds = dict(replay=t_replay, noise=t_noise, **t_csd, track=t_track)
    if "truth" in stages:
        yield at("truth")
    reference, t_ref = source.reference(tracking, seeds)
    if "truth" in stages:
        seconds["truth"] = t_ref
    yield at("score"); t0 = time.perf_counter(); M = source.regions.matrix(tg); s = source.score(M, reference); seconds["score"] = time.perf_counter() - t0
    seconds["total"] = sum(seconds.values())
    yield Result(protocol, meas, noisy, floor, s0_factor, sh, tg, seeds, M, s, snr=snr, physics=physics, clean=dwi,
                 seconds=seconds, reference=reference, extras=extras)


def run(source, protocol, *, snr=None, tracking=Tracking(), noise_seed=0, physics=None, prepared=None, progress=None):
    """The whole pipeline for one protocol: every stage's output and time. ``progress(stage, k, n)`` is called as
    each stage begins."""
    for item in run_stages(source, protocol, snr=snr, tracking=tracking, noise_seed=noise_seed, physics=physics, prepared=prepared):
        if isinstance(item, Result):
            return item
        if progress:
            progress(*item)


def repeat_tracking(res, source, tracking, keys):
    """The tractography of ``res`` (its FOD field, the seeds built once) repeated over tracker ``keys``: a generator
    of ``(key, matrix, score, seconds)`` scored against ``res``'s own reference; the replay, noise and
    reconstruction are kept, only the tracker's randomness varies."""
    fld, seeds = source.tracking_inputs(res.sh, tracking.density)
    for k in keys:
        t0 = time.perf_counter()
        tg, _ = track(fld, seeds, replace(tracking, key=int(k)), source.backend)
        M = source.regions.matrix(tg)
        yield int(k), M, source.score(M, res.reference), time.perf_counter() - t0
