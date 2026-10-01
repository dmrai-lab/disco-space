"""The brain source: a brain composed, voxel by voxel, from a white-matter FOD field, WM / GM / CSF fractions and
replay packs, replayed on the device, reconstructed, tracked and scored against the connectome of its own input.

The input is an **asset** (the directory layout of ``SubstrateCommons/masivar-brain``: ``manifest.json``,
``fod_wm.npy``, ``fractions.npy``, ``mask.npy``, ``labels.npy``, ``regions.json``, ``stop_mask.npy``,
``mean_b0.npy``): the multi-tissue CSD of a real scan, made once offline, with its parcellation. The replay is the
kernel route of dmipy-sim's phantom (RPH.md 6): per voxel and measurement

    S = | f_wm m0_wm FOD . C_wm  +  f_gm m0_gm E_gm  +  f_csf m0_csf E_csf |

where ``C_wm`` is the WM pack's pose response contracted to the FOD's harmonics
(:meth:`dmipy_sim.replay.ReplayPack.pose_responses` at ``keep=(8, 0)`` through :func:`dmipy_sim.replay.so3.axis_density_coeffs`),
``E_gm`` the GM pack's response to an isotropic orientation distribution, ``E_csf`` free water in closed form
(:class:`dmipy_sim.phantom.FreeWater`), each at the run's tissue (T2 / T1 per pool at the echo, the WM walls' surface
relaxivity, the myelin sheath's susceptibility at the scanner's field and its direction), and ``m0`` the proton
density per tissue. The pose responses (:meth:`Brain.prepare`: seconds on the CPU) are cached in the page's process per
pack, protocol, tissue and field; a run looks them up on the page and computes the missing ones inside the GPU call,
so the call is entered as soon as the request arrives. The contraction over voxels x
measurements x 45 runs in torch on the device. It is exactly what ``Phantom.compose(...).replay(seq, pose=...)``
computes, which the tests check. The acquisition is in the image frame; the field's direction is given in the
scanner frame and reaches the packs through the specimen's pose (:func:`specimen_pose`).
"""
from __future__ import annotations

import functools
import json
import os
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .. import pipeline as P

ASSET_FILES = ("manifest.json", "fod_wm.npy", "fractions.npy", "mask.npy", "labels.npy", "regions.json", "stop_mask.npy", "mean_b0.npy")
TISSUES = ("wm", "gm", "csf")
WM_POOLS = ("intra", "extra", "myelin")                  # the catalogue's white-matter pools, by the packs' pool names
M0_KEYS = {"wm": "proton_density_white_matter", "gm": "proton_density_grey_matter", "csf": "proton_density_csf"}
RELAXATION_KEYS = {"gm": ("T2_grey_matter", "T1_grey_matter"), "csf": ("T2_csf", "T1_csf")}
SCAN = "scan"                                             # the timing class of the asset's own protocol
B0_MODES = {"along the bore (z)": P.B0_ALONG_Z, "transverse (x): 90° from the bore, as in a biplanar magnet like the Swoop": P.B0_TRANSVERSE}
FREE_B0 = "free (polar and azimuth angles below)"
LMAX = 8
N_COEF = 45
REACH_MM = 250.0                                          # the longest streamline the tracker must be able to reach
PROFILE_ANGLES = np.linspace(0.0, 90.0, 19)               # degrees between the gradient and a fibre, for the response plot
C00 = 1.0 / (2.0 * np.sqrt(np.pi))                        # the l = 0 coefficient of a unit-integral density


# ---- the asset ---------------------------------------------------------------------------------------------------------

def asset_location(cfg):
    """Where the asset is: ``DISCO_BRAIN_ASSET`` in the environment (a local directory), else the configuration's
    ``[asset]``: ``local`` (a directory) or ``hub`` + ``revision`` (a public dataset downloaded into the Hub cache)."""
    env = os.environ.get("DISCO_BRAIN_ASSET")
    if env:
        return env
    a = cfg["asset"]
    return a["local"] if a.get("local") else f"hf://{a['hub']}@{a['revision']}"


def asset_dir(where):
    """The asset's local directory: ``where`` itself, or the dataset ``hf://repo@revision`` downloaded (once per
    container: the Hub cache holds it)."""
    if not where.startswith("hf://"):
        return where
    from huggingface_hub import snapshot_download
    repo, _, rev = where[len("hf://"):].partition("@")
    return snapshot_download(repo, repo_type="dataset", revision=rev or None, allow_patterns=list(ASSET_FILES))


@functools.lru_cache(maxsize=4)
def manifest(where):
    """The asset's ``manifest.json`` (downloaded alone for a Hub asset)."""
    if where.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        repo, _, rev = where[len("hf://"):].partition("@")
        path = hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=rev or None)
    else:
        path = os.path.join(where, "manifest.json")
    with open(path) as f:
        return json.load(f)


@functools.lru_cache(maxsize=16)
def declared_segments(uri):
    """The segment table ``{n, n_t, T}`` (RPK.md 4.3) a pack declares, read without its arrays: a Hub pack's
    record in its dataset's ``manifest.json`` (``hf://owner/name/path``), a local pack's header."""
    from dmipy_sim.replay.publish import MANIFEST, header_of, is_hub_uri, parse_uri
    if is_hub_uri(uri):
        from huggingface_hub import hf_hub_download
        repo, path = parse_uri(uri)
        with open(hf_hub_download(repo, MANIFEST, repo_type="dataset")) as f:
            rows = [r for r in json.load(f).get("packs") or [] if r.get("path") == path]
        if not rows:
            raise ValueError(f"{uri}: the dataset {repo} holds no record of this pack (its manifest.json)")
        seg = rows[0].get("segments")
    else:
        seg = header_of(uri)["walk_params"].get("segments")
    if not seg:
        raise ValueError(f"{uri}: no declared segment table (walk_params.segments, RPK.md 4.3) in its "
                         f"{'record' if is_hub_uri(uri) else 'header'}")
    return dict(n=int(seg["n"]), n_t=int(seg["n_t"]), T=float(seg["T"]))


def image_rotation(man):
    """``R`` (image -> scanner) of the asset's grid (:meth:`dmipy_sim.phantom.Grid.from_oblique_affine`)."""
    from dmipy_sim.phantom import Grid
    _, R = Grid.from_oblique_affine(np.asarray(man["grid"]["affine"], np.float64), tuple(man["grid"]["shape"]))
    return np.asarray(R, np.float64)


def read_regions(path):
    """``(regions, groups)`` from ``regions.json``: the regions ``[{id, name, hemisphere, lobe}]`` (a list, or the
    ``regions`` of an object) sorted by id, and the lobar groups ``((name, (id, ...)), ...)`` made from each region's
    hemisphere and lobe in order of first appearance; an object's ``groups`` must agree with them."""
    with open(path) as f:
        doc = json.load(f)
    regions = sorted(doc["regions"] if isinstance(doc, dict) else doc, key=lambda r: int(r["id"]))
    groups = {}
    for r in regions:
        groups.setdefault(f"{str(r['hemisphere'])[0].upper()} {r['lobe']}", []).append(int(r["id"]))
    groups = tuple((name, tuple(ids)) for name, ids in groups.items())
    if isinstance(doc, dict) and doc.get("groups"):
        stated = {tuple(sorted(int(i) for i in g["ids"])) for g in doc["groups"]}
        if stated != {tuple(sorted(ids)) for _, ids in groups}:
            raise ValueError("regions.json: its groups are not the regions' hemisphere x lobe")
    return regions, groups


@dataclass
class Asset:
    """The asset, cropped to the brain's bounding box (one voxel of margin): ``fod`` ``(X, Y, Z, 45)`` in the image
    frame, ``fractions`` ``(X, Y, Z, 3)`` WM / GM / CSF (a voxel whose fractions sum above one is divided by its sum, a
    WM fraction without an FOD is not replayed; ``notes`` counts both), ``mask``, ``labels``, ``stop``, ``b0``, ``affine`` the
    cropped grid's voxel -> scanner mm, ``R`` image -> scanner, ``regions`` / ``groups`` from ``regions.json``,
    ``manifest``, ``crop`` the slices into the full grid."""
    manifest: dict
    fod: np.ndarray
    fractions: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    stop: np.ndarray
    b0: np.ndarray
    affine: np.ndarray
    R: np.ndarray
    regions: list
    groups: tuple
    crop: tuple
    notes: tuple = ()

    @classmethod
    def load(cls, where):
        d = asset_dir(where)
        missing = [f for f in ASSET_FILES if not os.path.exists(os.path.join(d, f))]
        if missing:
            raise FileNotFoundError(f"the brain asset at {where} lacks {missing}")
        man = manifest(where)
        shape = tuple(man["grid"]["shape"])
        fod = np.load(os.path.join(d, "fod_wm.npy"), mmap_mode="r")
        mask = np.load(os.path.join(d, "mask.npy")).astype(bool)
        if fod.shape != shape + (N_COEF,) or mask.shape != shape:
            raise ValueError(f"the asset's FOD {fod.shape} and mask {mask.shape} are not on its grid {shape} with {N_COEF} coefficients")
        from dmipy_sim.phantom.orientation import SH_BASES
        basis = str(man["fod"]["basis"]).split()[0]
        if SH_BASES.get(basis) != ("tournier07", False) or int(man["fod"]["lmax"]) != LMAX or man["fod"]["frame"] != "image":
            raise ValueError(f"the asset's FOD is {man['fod']}: this source reads dmipy-sim's required basis (a basis whose "
                             f"name, the first word, dmipy_sim.phantom.orientation.SH_BASES maps to orthonormal tournier07: "
                             f"tournier07, mrtrix3, dmipy-fit) at lmax {LMAX} in the image frame")
        if list(man["fractions"]) != list(TISSUES):
            raise ValueError(f"the asset's fractions are {man['fractions']}, this source reads {list(TISSUES)}")
        ijk = np.argwhere(mask)
        lo = np.maximum(ijk.min(0) - 1, 0); hi = np.minimum(ijk.max(0) + 2, shape)
        crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        affine = np.asarray(man["grid"]["affine"], np.float64).copy()
        affine[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
        regions, groups = read_regions(os.path.join(d, "regions.json"))
        labels = np.load(os.path.join(d, "labels.npy"))[crop].astype(np.int32)
        n = int(man["parcellation"]["n_regions"])
        if [int(r["id"]) for r in regions] != list(range(1, n + 1)) or labels.min() < 0 or labels.max() > n:
            raise ValueError(f"the asset's regions are not 1..{n} (regions.json ids {[r['id'] for r in regions][:5]}..., labels up to {labels.max()})")
        m = mask[crop]
        fodc = np.asarray(fod[crop], np.float32) * m[..., None]
        fr = np.clip(np.load(os.path.join(d, "fractions.npy"))[crop].astype(np.float32), 0.0, None) * m[..., None]
        notes = []
        no_fod = (fr[..., 0] > 0) & ~(fodc[..., 0] > 0)
        if no_fod.any():                                  # a WM fraction needs an orientation to compose with
            notes.append(("voxels with a WM fraction and no FOD (their WM fraction is not replayed)",
                          f"{int(no_fod.sum()):,}, WM fraction up to {float(fr[..., 0][no_fod].max()):.3f}"))
            fr[..., 0] = np.where(no_fod, 0.0, fr[..., 0])
        total = fr.sum(-1, keepdims=True)
        over = total[..., 0] > 1.0
        if over.any():                                    # shares of a voxel's volume: float16 rounding, an unconstrained fit
            notes.append(("voxels whose fractions sum above one (divided by their sum)", f"{int(over.sum()):,}, largest sum {float(total.max()):.3f}"))
            fr = fr / np.maximum(total, 1.0)
        return cls(man, fodc, fr, m, labels, np.load(os.path.join(d, "stop_mask.npy"))[crop].astype(bool) & m,
                   np.load(os.path.join(d, "mean_b0.npy"))[crop].astype(np.float32), affine, image_rotation(man), regions, groups, crop, tuple(notes))


def scan_protocol(man, *, name):
    """The asset's own protocol: its rows' b-values (s/mm^2) and directions turned into the image frame (the frame
    of the FOD the replay composes), one shell per b rounded to 50 s/mm^2, on the timing class :data:`SCAN`."""
    b = np.asarray(man["protocol"]["bvals_s_mm2"], np.float64)
    g = np.asarray(man["protocol"]["bvecs"], np.float64)
    if g.shape == (3, len(b)):
        g = g.T
    frame = man["protocol"]["bvec_frame"]
    if frame not in ("image", "scanner"):
        raise ValueError(f"bvec_frame is 'image' or 'scanner', got {frame!r}")
    if frame == "scanner":
        g = g @ image_rotation(man)                      # R.T g per row
    group = np.round(b / 50.0).astype(int)
    return P.protocol_from_rows(b, g, group, lambda k, rows: P.Shell(SCAN, float(np.round(b[rows].mean())), int(len(rows))), name=name)[0]


# ---- the physics --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BrainPhysics:
    """The tissue and scanner a brain replay is evaluated at: ``field_T`` (the scanner's field, T; the catalogue's
    relaxation is cited at the nearest of 1.5 / 3 / 7 T), ``b0_direction`` (its unit direction in the scanner frame),
    ``T2`` / ``T1`` as ``{"wm": {pool: s}, "gm": s, "csf": s}``, ``rho`` the WM walls' surface relaxivity (m/s),
    ``chi_iso`` / ``chi_aniso`` the myelin sheath's susceptibility (SI, the WM pack's field source), ``D_csf`` free
    water's diffusivity (m^2/s), ``m0`` the proton density per tissue relative to CSF, ``wm_pack`` / ``gm_pack`` the
    packs by their configuration label; ``relaxation``, ``contact`` and ``field`` switch the three tiers. All off is
    bare diffusion at the same proton densities and packs."""
    field_T: float
    T2: dict
    T1: dict
    rho: float
    chi_iso: float
    chi_aniso: float
    D_csf: float
    m0: dict
    wm_pack: str
    gm_pack: str
    b0_direction: tuple = P.B0_ALONG_Z
    relaxation: bool = True
    contact: bool = True
    field: bool = True

    def __post_init__(self):
        if not (0 < self.field_T < 30):
            raise ValueError(f"the field is in tesla, got {self.field_T}")
        if set(self.m0) != set(TISSUES) or any(float(v) < 0 for v in self.m0.values()):
            raise ValueError(f"m0 is a non-negative proton density per tissue {TISSUES}, got {self.m0}")
        if self.rho < 0:
            raise ValueError("the surface relaxivity is non-negative")
        u = np.asarray(self.b0_direction, np.float64)
        if u.shape != (3,) or not np.isclose(np.linalg.norm(u), 1.0, atol=1e-6):
            raise ValueError("b0_direction is a unit vector")

    @property
    def bare(self):
        return not (self.relaxation or self.contact or self.field)

    @property
    def scanner(self):
        """The replay's ``scanner=``: the field in tesla with the field tier, else None (relaxation at a field is
        already in T2 and T1)."""
        return self.field_T if self.field else None

    def label(self):
        if self.bare:
            return "bare diffusion"
        tiers = "+".join(t for t in ("relaxation", "contact", "field") if getattr(self, t))
        u = self.b0_direction
        return f"{self.field_T:g} T along ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f}), tiers {tiers}"

    def tissues(self, wm_pools, gm_pools):
        """``(wm, gm, csf)`` :class:`dmipy_sim.spec.Tissue`: the WM pack's pools ``wm_pools`` with their T2 / T1, the
        walls' rho and the sheath's susceptibility, the GM pack's pools ``gm_pools`` at grey matter's T2 / T1, free
        water's D, T2 and T1; a tier off leaves its values out (``None`` for a pack with nothing on: its bare
        diffusion)."""
        from dmipy_sim.spec import Tissue
        relax, contact, field = self.relaxation, self.contact, self.field
        wm = Tissue(T2={p: self.T2["wm"][p] for p in wm_pools} if relax else None, T1={p: self.T1["wm"][p] for p in wm_pools} if relax else None,
                    rho=self.rho if contact else None, chi_iso=self.chi_iso if field else None,
                    chi_aniso=self.chi_aniso if field else 0.0) if (relax or contact or field) else None
        gm = Tissue(T2={p: self.T2["gm"] for p in gm_pools}, T1={p: self.T1["gm"] for p in gm_pools}) if relax else None
        csf = Tissue(D=self.D_csf, T2=self.T2["csf"] if relax else None, T1=self.T1["csf"] if relax else None)
        return wm, gm, csf


def catalogue(field_T):
    """The catalogue at the cited field nearest ``field_T`` (in log distance): the white matter's pools
    (:func:`space.pipeline.catalogue`), grey matter's and CSF's T2 and T1, free water's diffusivity, the proton
    densities, all from ``dmipy_sim.substrate.biophysical_constants`` by key; ``catalogue_field`` says which field."""
    import warnings
    from dmipy_sim.substrate.biophysical_constants import get_value
    wm = P.catalogue(field_T, WM_POOLS)
    near = wm["catalogue_field"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T2 = {"wm": wm["T2"], **{t: float(get_value(k[0], near, allow_nearest=True)) for t, k in RELAXATION_KEYS.items()}}
        T1 = {"wm": wm["T1"], **{t: float(get_value(k[1], near, allow_nearest=True)) for t, k in RELAXATION_KEYS.items()}}
        return dict(catalogue_field=near, T2=T2, T1=T1, rho=wm["rho"], chi_iso=wm["chi_iso"], chi_aniso=wm["chi_aniso"],
                    D_csf=float(get_value("D_csf")), m0={t: float(get_value(k)) for t, k in M0_KEYS.items()})


def specimen_pose(R, b0_direction):
    """The specimen's pose in the bore for a replay whose acquisition is in the image frame: ``W R`` with ``R`` the
    image -> scanner rotation and ``W`` the rotation taking the field's scanner-frame direction onto z (the bore's
    field axis), so the packs see the field at ``R.T b0_direction`` in the image frame. A sequence composed at this
    pose carries the gradients ``pose @ g`` (:meth:`Brain.sequence`)."""
    from dmipy_sim.replay.so3 import rotation_of
    return rotation_of(np.asarray(b0_direction, np.float64)).T @ np.asarray(R, np.float64)


@dataclass
class Kernels:
    """A replay's pose responses (:meth:`Brain.prepare`): per measurement, the WM response contracted to the FOD's
    harmonics ``wm`` ``(n_meas, 45)`` complex, the GM response to an isotropic distribution ``gm`` and free water's
    ``csf`` ``(n_meas,)`` complex, the proton densities ``m0``, the packs' certified floors, the seconds it took;
    ``profile`` the true response curves at the run's shells (``b``, ``wm`` ``(n_shells, n_angles)``, ``gm``, ``csf``
    ``(n_shells,)``, each over its b = 0 value) and ``b0`` the b = 0 signal per tissue under each tier alone."""
    wm: np.ndarray
    gm: np.ndarray
    csf: np.ndarray
    m0: dict
    floors: dict
    seconds: float
    profile: Optional[dict] = None
    b0: Optional[dict] = None


# ---- the source ---------------------------------------------------------------------------------------------------------

class Brain(P.Source):
    """A brain from its asset: the regions of its parcellation, the tracker seeded in its white matter and stopped
    where the white matter and the regions end, the connectome scored against the connectome of the asset's own FOD
    tracked with the same settings, seeds and key."""
    TIERS = ("relaxation", "contact", "field")
    STAGES = ("replay", "noise", "csd", "track", "truth", "score")

    def __init__(self, cfg, *, asset=None):
        super().__init__(cfg)
        self.where = asset or asset_location(cfg)
        self.asset = a = Asset.load(self.where)
        self.shapes = self.shape_table(cfg, self.where)
        self.affine = a.affine
        self.mask = a.mask
        self.regions = P.Regions(a.labels, a.affine, tuple(r["name"] for r in a.regions), a.groups)
        self.vox = np.argwhere(a.mask)                                   # the replayed voxels
        f = a.fractions[a.mask].astype(np.float64)
        c0 = a.fod[a.mask][:, 0].astype(np.float64)
        unit = np.where((c0 > 0)[:, None], a.fod[a.mask] * (C00 / np.where(c0 > 0, c0, 1.0))[:, None], 0.0)
        self.fod_unit = unit.astype(np.float32)                           # every voxel's FOD as a unit-integral density
        self.f = f.astype(np.float32)
        self.packs = {}
        self._reference = {}
        self._kernels = {}                                                # prepare()'s results, by response_key
        self._compiled = set()                                            # the _band()s prepare() ran in this process

    @classmethod
    def load(cls, cfg, *, local=None):
        return cls(cfg, asset=local)

    def warm_entries(self):
        """The ``(meas, physics)`` a run can ask for without a custom acquisition, in the order the warm-up
        computes them: every preset at the panel's defaults with its ladder rungs, then B of every knob on the
        default protocol, the field presets along both B0 directions, the pulse-timing knobs last (each a band of
        its own to compile, the longest reaching a second window of the packs). An entry the packs cannot play is
        left out (a run asking for it is refused by name)."""
        cfg = self.cfg
        panel = self.panel(cfg)
        values = {c.name: c.value for row in panel.rows for c in row if c.name in panel.fields}
        physics = self.physics_from(cfg, values)
        entries = []
        for name in self.presets(cfg):
            prot = self.protocol(cfg, name)
            entries += [(prot, physics)] + [(prot, rung) for _, rung in self.ladder_steps(physics)]
        scan = self.protocol(cfg, self.presets(cfg)[0])
        knobs = [c for c in self.knobs(cfg).values() if c is not None]
        for change in [c for c in knobs if c[0] != "shape"] + [c for c in knobs if c[0] == "shape"]:
            prot, _, _, v = self.apply_knob(cfg, change, scan, True, None, values)
            modes = list(B0_MODES) if change[0] == "field" else [v["b0_mode"]]
            entries += [(prot, self.physics_from(cfg, {**v, "b0_mode": mode})) for mode in modes]
        out = []
        for prot, ph in entries:
            try:
                out.append((self.validate(prot, ph), ph))
            except ValueError:
                continue
        return out

    def warm(self):
        """The packs loaded (the windows :meth:`warm_entries` reach, downloaded once per container) and every entry's
        pose responses computed and kept in this process, in order: what a visitor would otherwise pay inside the
        GPU reservation. The bands it compiles stay compiled in this process and in every worker forked from it."""
        for meas, ph in self.warm_entries():
            if self.cached(meas, ph) is None:
                self.prepare(meas, ph)

    def warm_in_background(self):
        """:meth:`warm` in a process of its own while this one serves: the entries' responses arrive one by one and
        are kept here (:meth:`keep`), so a run finds the cache filling from the first entry on and pays, inside its
        reservation, only for an entry not yet there. A spawned process, not a thread: the GPU worker is a fork of
        this process, and a fork taken while a thread runs JAX can deadlock the child; the spawned process compiles
        its own bands, so an entry of a band this process never compiled is priced cold (:meth:`responses`) until a
        run computes it here. Returns the thread that drains the results; the packs are loaded here first (the
        windows the entries reach), so a run arriving before the warm-up ends has them."""
        import multiprocessing as mp
        import threading
        entries = self.warm_entries()
        for meas, ph in entries:
            for tissue, label in (("wm", ph.wm_pack), ("gm", ph.gm_pack)):
                self.pack(tissue, label, self.windows_needed(tissue, label, meas))
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_warm_worker, args=(self.cfg, self.where, entries, queue), daemon=True, name="brain-warm")
        proc.start()

        def drain():
            while True:
                item = queue.get()
                if item is None:
                    break
                key, kernels = item
                self.keep({key: kernels})
            proc.join()

        t = threading.Thread(target=drain, daemon=True, name="brain-warm-drain")
        t.start()
        return t

    # ---- the packs ----
    @classmethod
    def pack_menu(cls, cfg, tissue):
        """The configuration's packs for ``tissue`` (``wm`` / ``gm``): ``{label: uri}`` in the menu's order."""
        return {p["label"]: p["uri"] for p in cfg["packs"][tissue]}

    def segments(self, tissue, label):
        """The declared segment table ``{n, n_t, T}`` of the pack ``label`` of ``tissue``'s menu
        (:func:`declared_segments`: read without the pack's arrays)."""
        menu = self.pack_menu(self.cfg, tissue)
        if label not in menu:
            raise ValueError(f"no {tissue.upper()} pack {label!r}; the menu is {list(menu)}")
        return declared_segments(menu[label])

    def windows_reached(self, segments, meas):
        """``{class: k}``: per timing class of ``meas``, the number of the pack's windows its acquisition reaches.
        ``segments`` is a declared table ``{n, n_t, T}`` (:func:`declared_segments`) or a pack's meta (its
        ``walk_params.segments``). The acquisition lasts ``T_acq = (G.shape[1] - 1) dt`` of the class's sequence,
        as :meth:`dmipy_sim.replay.ReplayPack._compile` reads it, and window ``i`` starts at ``i T``; a window whose
        first save sits at or beyond ``T_acq`` is not reached, so the save at a window's end belongs to that window
        alone. A pack of one window (``n = 1``) is reached as one window whatever the class; ``k``
        above ``n`` is an acquisition beyond the walk (:meth:`windows_needed` refuses it)."""
        seg = (segments.get("walk_params") or {}).get("segments") if "walk_params" in segments else segments
        out = {}
        for name in map(str, np.unique(meas.shape)):
            if int(seg["n"]) <= 1:
                out[name] = 1
                continue
            s = self._b0_sequence(name, np.eye(3))
            T_acq = (np.shape(s.G)[1] - 1) * float(s.dt)
            out[name] = max(1, int(np.ceil(T_acq * (1.0 - 1e-12) / float(seg["T"]))))
        return out

    def windows_needed(self, tissue, label, meas):
        """The windows of the pack ``label`` of ``tissue``'s menu the classes of ``meas`` reach (the largest
        :meth:`windows_reached`), refused by name for a class that reaches beyond the windows the pack declares."""
        seg = self.segments(tissue, label)
        reach = self.windows_reached(seg, meas)
        for name, k in reach.items():
            if k > int(seg["n"]):
                raise ValueError(f"the {tissue.upper()} pack {label!r} declares {seg['n']} window(s) of {float(seg['T']) * 1e3:g} ms: "
                                 f"the class {name!r} (TE {float(self.shapes[name]['TE']) * 1e3:g} ms) reaches window {k - 1}")
        return max(reach.values())

    @staticmethod
    def saves_spanned(segments, k):
        """The saves of the first ``k`` windows of a declared table: ``k (n_t - 1) + 1`` (consecutive windows share
        their boundary save; one window is its ``n_t``)."""
        return int(k) * (int(segments["n_t"]) - 1) + 1

    def pack(self, tissue, label, k):
        """The pack ``label`` of ``tissue``'s menu holding at least its first ``k`` windows: a Hub URI whose
        declared table has ``n > 1`` loads ``windows=range(k)`` (:class:`dmipy_sim.phantom.PackSubstrate`; the pack
        records them as ``windows_present``); a local path or a one-window pack loads whole. A later request for
        more windows than it holds reloads with the larger range (the library fetches only the new ranges) and
        replaces the cached pack."""
        from dmipy_sim.phantom import PackSubstrate
        from dmipy_sim.replay.publish import is_hub_uri
        key = (tissue, label)
        seg = self.segments(tissue, label)
        if key in self.packs and self.windows_held(self.packs[key], seg) >= int(k):
            return self.packs[key]
        uri = self.pack_menu(self.cfg, tissue)[label]
        windows = {"windows": range(int(k))} if is_hub_uri(uri) and int(seg["n"]) > 1 else {}
        self.packs[key] = PackSubstrate(uri, m0=1.0, name=f"{tissue}:{label}", **windows).pack
        return self.packs[key]

    @staticmethod
    def windows_held(pack, segments):
        """The windows ``pack`` holds: its ``windows_present`` when loaded by window, else every window ``segments``
        declares."""
        return int(segments["n"]) if pack.windows_present is None else int(pack.windows_present)

    @staticmethod
    def pools(pack):
        """The pack's pool names (its embedded spec), or None for a pack without a spec."""
        spec = pack.substrate
        return None if spec is None else tuple(p.name for p in spec.pools)

    @staticmethod
    def walk_seconds(pack):
        return float(pack.meta["walk_params"]["T_max"])

    # ---- the page's side ----
    @classmethod
    def shape_table(cls, cfg, where=None):
        """The timing classes: the configuration's ``[shapes]`` and :data:`SCAN`, the scan's own pulse timing from the
        asset's manifest, or the configuration's ``[scan]`` timing when the manifest states none (the label says
        which)."""
        pr = manifest(where or asset_location(cfg))["protocol"]
        stated = all(pr.get(k) is not None for k in ("delta_s", "Delta_s", "TE_s"))
        t = dict(delta=float(pr["delta_s"]), Delta=float(pr["Delta_s"]), TE=float(pr["TE_s"])) if stated else \
            {k: float(cfg["scan"][k]) for k in ("delta", "Delta", "TE")}
        why = "the scan's own timing" if stated else "the scan's b-values and directions at the configuration's timing (the asset states none)"
        return {**cfg["shapes"], SCAN: dict(label=f"{why}: δ {t['delta'] * 1e3:g} / Δ {t['Delta'] * 1e3:g} ms, TE {t['TE'] * 1e3:g} ms", **t)}

    @classmethod
    def describe(cls, cfg):
        d = cfg["describe"]
        return dict(
            title=d["title"],
            heading=(f"# {d['title']}\n" + d["heading"]),
            acquisition=("A shell's **pulse timing** (δ, Δ, TE; square pulses) is one of the classes below, each played by the packs' "
                         "walks, stored in windows of 100 ms: a class reads the windows its acquisition reaches, and a timing beyond the "
                         "walk is refused by name; the b-values, the directions and their number, the "
                         "tissue, the proton densities, the packs, the SNR and the tracker are free. The first preset is the scan's own "
                         "protocol, its directions turned into the image frame. Timing classes: "
                         + "; ".join(f"`{n}` = {t['label']}" for n, t in cls.shape_table(cfg).items()) + "."),
            tissue=("The replay composes three tissues per voxel: the WM pack contracted with the voxel's FOD, the GM pack as an "
                    "isotropic distribution, free water in closed form, each weighted by its fraction and proton density M0 "
                    "(relative to CSF). **SNR is defined at M0 = 1**, the b = 0 signal of a pure-CSF voxel before relaxation, so "
                    "lowering a tissue's M0 lowers its SNR. The **field** tier is the WM pack's myelin-sheath susceptibility at the "
                    "scanner's field and direction (given in the scanner frame; the head's own susceptibility is not modelled); the "
                    "GM pack and free water carry no field source. The packs' responses to the presets and the field presets are "
                    "computed once when the Space starts; any other (a custom protocol, a field or tissue not seen before) is "
                    "computed inside the GPU call and is part of what the run reserves. The pack menu is the "
                    "configuration's list; the WM pack's contact tier uses the catalogue's white-matter ρ, the GM pack's walls take "
                    "none (no cited value)."),
            explorer=("What the replay made, before the noise and the tractography. **A** is the run you configured (its replay "
                      "before the noise); **B** is the same run with the one knob you chose changed. **Ingredients**: the proton-density "
                      "map Σ f·M0, the relaxation weight of every voxel's b = 0 signal at the echo, and the contact tier's survival "
                      "(the field tier's effect is its rung of the ladder). "
                      "**Layers**: A replayed with its tiers switched on one at a time (bare → + relaxation → + contact → + field), then B, as the "
                      "DWI or the tensor's MD and FA. **Estimated vs true response**: the responses the reconstruction estimated from the "
                      "replayed data against the packs' exact response at the run's tissue, per shell: what the response heuristic "
                      "gets right is itself a result. There is no per-voxel split-half floor here: the accuracy is the packs' "
                      "certified floors (the accuracy table)."),
            results_tab="4 · brain results",
            truth_tab="2 · the input",
            truth_labels=("the regions (centroids, by lobe)", "the input: WM FOD peaks, fractions and regions"),
            stages={"replay": "replaying the brain from the packs' responses", "noise": "adding Rician noise",
                    "csd": "estimating the responses and fitting CSD", "track": "tracking from the white matter",
                    "truth": "tracking the input FOD with the same settings (the truth)", "score": "scoring the connectome"},
            fixed=[f"the subject: {d['subject']}; its FOD field, fractions and parcellation (made once, offline)",
                   "one pack per tissue class per run, from the configuration's menu (Hub records later)",
                   "the head's own susceptibility and RF effects are not modelled",
                   "no per-voxel split-half floor: the packs' certified floors are the accuracy"],
            accuracy=("A brain replay is exact composition (the tests check it against dmipy-sim's own `Phantom.compose(...).replay`) of "
                      "the packs' responses, so its accuracy is the packs' certified Monte-Carlo floors; there is no per-voxel split-half "
                      "floor, because every voxel reads the same two walks."),
            ingredients=("proton density: Σ f·M0 (relative to CSF)", "relaxation: the b = 0 signal's T2 / T1 weight at the echo",
                         "contact: the WM walls' surface-relaxivity survival at b = 0"),
            truth=("the connectome of the input FOD tracked directly with the same tracker, settings, seeds and key: Pearson of "
                   "log(1 + count) over the region pairs, the pairs connected in one only, and the same on the lobar groups"),
            ladder="Replay DWI Explorer: replay A's tier ladder too (bare, + relaxation, + contact; noise-free; a few seconds of GPU time)",
            views=dict(truth=True, fractions=True, lobar=True, roundtrip=True, response=True, floor=False),
            files="brain", step_unit="mm",
            volumes_label="DWI (.nii.gz, bvecs in the image frame) with bvals/bvecs, the FOD SH field (tournier07 order 8), the connectome (.csv)",
            run_label="replay → responses + CSD → track → connectome")

    @classmethod
    def presets(cls, cfg):
        return [cfg["describe"]["scan_preset"]] + list(cfg["presets"])

    @classmethod
    def protocol(cls, cfg, name):
        if name == cfg["describe"]["scan_preset"]:
            return scan_protocol(manifest(asset_location(cfg)), name=name)
        return P.preset_protocol(cfg, name)

    @classmethod
    def panel(cls, cfg):
        phys = cfg["physics"]; f0 = float(phys["default_field"]); c0 = cls.catalogue_numbers(cfg, f0)
        cat = catalogue(f0)
        C = P.Control
        wm_menu = list(cls.pack_menu(cfg, "wm")); gm_menu = list(cls.pack_menu(cfg, "gm"))
        rows = (
            (C("on", "checkbox", "evaluate in tissue (off: bare diffusion at the same proton densities)", bool(phys["default_on"])),),
            (C("field_preset", "field_preset", "field preset", f"{f0:g} T", tuple(f"{float(f):g} T" for f in phys["fields"])),
             C("field_T", "slider", "B0 (T)", f0, minimum=0.05, maximum=12.0, step=0.001)),
            (C("b0_mode", "dropdown", "B0 direction (scanner frame)", list(B0_MODES)[0], tuple(B0_MODES) + (FREE_B0,)),
             C("theta", "slider", "polar angle from z (°)", 0, minimum=0, maximum=180, step=1),
             C("phi", "slider", "azimuth from x (°)", 0, minimum=0, maximum=360, step=1)),
            (C("relaxation", "checkbox", "relaxation (T2, T1 per tissue)", True), C("contact", "checkbox", "contact (the WM walls' ρ)", True),
             C("field", "checkbox", "field (the myelin sheath's susceptibility)", True)),
            tuple(C(f"T2_wm_{p}", "number", f"T2 WM {p} (ms)", c0[k]) for k, p in enumerate(WM_POOLS)),
            (C("T2_gm", "number", "T2 GM (ms)", c0[3]), C("T2_csf", "number", "T2 CSF (ms)", c0[4])),
            (C("rho", "number", "ρ of the WM walls (µm/s)", c0[5]), C("chi_iso", "number", "χ_iso of the sheath (ppm)", c0[6]),
             C("chi_aniso", "number", "Δχ_a of the sheath (ppm)", c0[7])),
            (C("m0_wm", "slider", "M0 WM (relative to CSF)", cat["m0"]["wm"], minimum=0.0, maximum=1.0, step=0.01),
             C("m0_gm", "slider", "M0 GM", cat["m0"]["gm"], minimum=0.0, maximum=1.0, step=0.01),
             C("m0_csf", "slider", "M0 CSF", cat["m0"]["csf"], minimum=0.0, maximum=1.0, step=0.01)),
            (C("wm_pack", "dropdown", "WM pack", wm_menu[0], tuple(wm_menu)), C("gm_pack", "dropdown", "GM pack", gm_menu[0], tuple(gm_menu))),
            (C("catalogue_note", "catalogue_note", value=c0[-1]), C("reset", "reset", "reset to the catalogue at this field")),
        )
        fields = ("on", "field_T", "b0_mode", "theta", "phi", "relaxation", "contact", "field", *[f"T2_wm_{p}" for p in WM_POOLS], "T2_gm", "T2_csf",
                  "rho", "chi_iso", "chi_aniso", "m0_wm", "m0_gm", "m0_csf", "wm_pack", "gm_pack")
        return P.Panel(rows, fields, (*[f"T2_wm_{p}" for p in WM_POOLS], "T2_gm", "T2_csf", "rho", "chi_iso", "chi_aniso"),
                       tuple(float(f) for f in phys["fields"]))

    @classmethod
    def catalogue_numbers(cls, cfg, field_T):
        """The catalogue's T2 per WM pool, GM and CSF (ms), the WM walls' rho (µm/s) and the sheath's susceptibility
        (ppm) at ``field_T``, and the note saying which cited field they came from."""
        c = catalogue(float(field_T))
        note = (f"catalogue values at {c['catalogue_field']:g} T" if abs(c["catalogue_field"] - float(field_T)) < 1e-9
                else f"the catalogue has no cited relaxation at {float(field_T):g} T: nearest is {c['catalogue_field']:g} T, edit as you see fit")
        return [c["T2"]["wm"][p] * 1e3 for p in WM_POOLS] + [c["T2"]["gm"] * 1e3, c["T2"]["csf"] * 1e3, c["rho"] * 1e6,
                                                          c["chi_iso"] * 1e6, c["chi_aniso"] * 1e6, note]

    @classmethod
    def physics_from(cls, cfg, values):
        """The panel (page units: ms, µm/s, ppm) as a :class:`BrainPhysics`: T1 and free water's D from the catalogue at
        the panel's field, the tiers off (bare diffusion at the same M0 and packs) when the panel is off."""
        on = bool(values["on"])
        c = catalogue(float(values["field_T"]))
        u = B0_MODES[values["b0_mode"]] if values["b0_mode"] in B0_MODES else P.b0_direction(values["theta"], values["phi"])
        return BrainPhysics(field_T=float(values["field_T"]), b0_direction=tuple(float(x) for x in u),
                            T2={"wm": {p: float(values[f"T2_wm_{p}"]) * 1e-3 for p in WM_POOLS}, "gm": float(values["T2_gm"]) * 1e-3, "csf": float(values["T2_csf"]) * 1e-3},
                            T1=c["T1"], rho=float(values["rho"]) * 1e-6, chi_iso=float(values["chi_iso"]) * 1e-6, chi_aniso=float(values["chi_aniso"]) * 1e-6,
                            D_csf=c["D_csf"], m0={t: float(values[f"m0_{t}"]) for t in TISSUES}, wm_pack=str(values["wm_pack"]), gm_pack=str(values["gm_pack"]),
                            relaxation=on and bool(values["relaxation"]), contact=on and bool(values["contact"]), field=on and bool(values["field"]))

    @classmethod
    def knobs(cls, cfg):
        """B = A with one change: the field (with the catalogue's tissue at it), the field's direction, a tier off, the
        tissue off, the WM or GM pack, grey matter's M0, the noise, or every shell's pulse timing (a stimulated echo
        among them)."""
        out = {P.NO_KNOB: None}
        for f in cfg["physics"]["fields"]:
            out[f"field → {float(f):g} T (catalogue tissue at that field)"] = ("field", float(f))
        for label in B0_MODES:
            out[f"B0 direction → {label}"] = ("b0", label)
        for tier in cls.TIERS:
            out[f"{tier} tier → off"] = ("tier", tier)
        out["tissue → off (bare diffusion)"] = ("bare", None)
        for tissue in ("wm", "gm"):
            for label in cls.pack_menu(cfg, tissue):
                out[f"{tissue.upper()} pack → {label}"] = ("pack", (tissue, label))
        for v in cfg["physics"]["m0_gm_knob"]:
            out[f"M0 of GM → {float(v):g}"] = ("m0_gm", float(v))
        for v in (10, 100):
            out[f"SNR → {v}"] = ("snr", float(v))
        out["noise → off"] = ("snr", None)
        for name, sh in cfg["shapes"].items():
            out[f"every shell's pulse timing → {name} ({sh['label']})"] = ("shape", name)
        return out

    @classmethod
    def apply_knob(cls, cfg, change, protocol, snr_on, snr, values):
        kind, value = change
        v = dict(values)
        if kind in ("field", "b0", "tier") and not v["on"]:
            raise ValueError(f"the knob {kind!r} changes the tissue panel, which is off for A: switch it on, or choose another knob")
        if kind == "field":
            v["field_T"] = value; v.update(zip(cls.panel(cfg).catalogue, cls.catalogue_numbers(cfg, value)))
        elif kind == "b0":
            v["b0_mode"] = value
        elif kind == "tier":
            v[value] = False
        elif kind == "bare":
            v["on"] = False
        elif kind == "pack":
            v[f"{value[0]}_pack"] = value[1]
        elif kind == "m0_gm":
            v["m0_gm"] = value
        elif kind == "snr":
            snr_on = value is not None; snr = value if value is not None else snr
        elif kind == "shape":
            protocol = P.retime(protocol, value)
        else:
            raise ValueError(f"unknown knob {kind!r}")
        return protocol, snr_on, snr, v

    @classmethod
    def tracking_controls(cls, cfg):
        t = cfg["tracking"]
        return dict(density=(1, 4, t["density"], 1, "seeds per WM voxel (density³)"), max_angle=(10, 60, t["max_angle"], 1, "max angle (°)"),
                    step=(0.5, 2.5, t["step_mm"], 0.05, "step (mm)"))

    @classmethod
    def tracking(cls, cfg, *, density, max_angle, step, key):
        """The tracker with :meth:`max_steps` for the step: a streamline can reach :data:`REACH_MM`."""
        return P.Tracking(density=int(density), step_mm=float(step), max_angle=float(max_angle), max_steps=cls.max_steps(step), key=int(key))

    @classmethod
    def shapes_of(cls, cfg):
        return cls.shape_table(cfg)

    @classmethod
    def max_steps(cls, step_mm):
        """Points per half-streamline such that :data:`REACH_MM` is reachable from a seed at either end."""
        return int(np.ceil(REACH_MM / float(step_mm))) + 1

    @classmethod
    def estimated_seconds(cls, cfg, protocol, *, density, knob, n_keys, ladder, responses=()):
        """The GPU seconds a run reserves: ``[budget]`` of the configuration, measured on the L40S with the BATMAN
        fixture (tools/measure_brain.py): a fixed part (the worker's start, noise, the reconstruction's fixed cost, the
        handoff), a part per measurement (the contraction and the reconstruction scale with the rows), the tracking
        and the truth's tracking per density, the ladder's two rungs of contraction, B (a second run whose truth is A's),
        each further tracker key, and the pose responses the worker computes because the page has them not cached
        (``responses``: ``(state, n_meas, saves)`` per entry, :meth:`responses`, each priced ``fixed + per_meas x n_meas +
        per_save x saves``); times the margin, within 30 and 480 s."""
        b = cfg["budget"]
        n = protocol.n_meas; d = str(int(density))
        track = float(b["track"][d])
        run = b["fixed"] + b["per_meas"] * n + track
        secs = run + track + (b["ladder_per_meas"] * n if ladder else 0.0) + ((run - b["worker"]) if knob != P.NO_KNOB else 0.0) + track * (int(n_keys) - 1)
        secs += sum(b[f"response_{state}"]["fixed"] + b[f"response_{state}"]["per_meas"] * m + b[f"response_{state}"]["per_save"] * saves
                    for state, m, saves in responses)
        return int(min(480, max(30, b["margin"] * secs)))

    # ---- the run's side ----
    def validate(self, protocol, physics):
        """The measurements, refused by name when a class reaches beyond the windows a pack declares
        (:meth:`windows_needed`, from the declared table, loaded or not), when a class's echo lies beyond a loaded
        pack's walk (:meth:`prepare` loads the chosen ones), or when the WM pack cannot carry the relaxation tier (a
        pack without a spec has no pool names to give T2 by)."""
        meas = P.measurements(protocol, self.shapes)
        if physics is None:
            raise ValueError("a brain replay needs its physics (the proton densities and the packs), even for bare diffusion")
        for tissue, label in (("wm", physics.wm_pack), ("gm", physics.gm_pack)):
            self.windows_needed(tissue, label, meas)
            pk = self.packs.get((tissue, label))
            if pk is None:
                continue
            T = self.walk_seconds(pk)
            for name in map(str, np.unique(meas.shape)):
                TE = float(self.shapes[name]["TE"])
                if TE > T * (1 + 1e-9):
                    raise ValueError(f"the {tissue.upper()} pack {label!r} is a {T * 1e3:g} ms walk: it cannot play the class {name!r} "
                                     f"at TE {TE * 1e3:g} ms")
            if tissue == "wm" and physics.relaxation and self.pools(pk) is None:
                raise ValueError(f"the WM pack {label!r} embeds no substrate spec, so its pools have no names to give T2 by "
                                 f"(RPK.md 8.5): switch the relaxation tier off or choose another WM pack")
            if tissue == "wm" and physics.field and not (pk.has_field or pk.field_is_zero):
                raise ValueError(f"the WM pack {label!r} stores no field channel: switch the field tier off or choose another WM pack")
        return meas

    def sequence(self, meas, rows, pose):
        """The ScannerSequence of the measurement rows ``rows`` (one timing class) played at the specimen's ``pose``:
        the image-frame directions turned by ``pose``, square pulses at the class's delta / Delta / TE, a stimulated
        echo storing for TM = Delta - delta when the class is one."""
        import dmipy_sim as d
        names = np.unique(meas.shape[rows])
        if len(names) != 1:
            raise ValueError(f"one timing class per sequence, got {list(names)}")
        t = self.shapes[str(names[0])]
        dirs = (meas.dirs[rows] @ np.asarray(pose).T).tolist(); b = meas.bvals[rows] * 1e6
        if t.get("kind", "pgse") == "pgste":
            return d.pgste(dirs, float(t["delta"]), float(t["Delta"]) - float(t["delta"]), bvalues=b, TE=float(t["TE"]), n_t=1000,
                           slew_rate=np.inf, ste_flip_angles=(90.0, 90.0, 90.0))
        return d.pgse(dirs, float(t["delta"]), float(t["Delta"]), bvalues=b, TE=float(t["TE"]), n_t=1000, slew_rate=np.inf)

    def _b0_sequence(self, name, pose):
        """The class ``name``'s b = 0 measurement as a sequence at ``pose`` (:meth:`sequence`): what the b = 0 signal
        and the acquisition's duration are read from."""
        one = P.Measurements(np.zeros(1), np.array([[0.0, 0.0, 1.0]]), np.array([name]), np.zeros(1), np.zeros(1), np.zeros(1))
        return self.sequence(one, np.arange(1), pose)

    def response_key(self, meas, physics=None):
        """The key of :meth:`prepare`'s result in this process's cache: the measurements (b-values, directions,
        timing classes) and the physics but its proton densities, which the device applies."""
        import hashlib
        h = hashlib.sha256()
        for x in (meas.bvals, meas.dirs, meas.shape.astype(str)):
            h.update(np.ascontiguousarray(x).tobytes())
        return h.hexdigest(), repr(replace(physics, m0={t: 0.0 for t in TISSUES}))

    @staticmethod
    def _band(meas, physics):
        """What the pose route compiles for: the packs, the rows per timing class and, with the field tier, the field
        and the sheath's susceptibility (the phase amplitude sets the expansion's band, whose shapes compile; measured:
        a new tissue or field direction at a compiled band costs no compilation, a run without the field tier none
        at all)."""
        names, counts = np.unique(meas.shape.astype(str), return_counts=True)
        field = (physics.field_T, physics.chi_iso, physics.chi_aniso) if physics.field else None
        return physics.wm_pack, physics.gm_pack, tuple(zip(names.tolist(), counts.tolist())), field

    def cached(self, meas, physics=None):
        """:meth:`prepare`'s result for ``meas`` at ``physics`` when this process has it (at ``physics``'s proton
        densities, zero seconds), else None."""
        k = self._kernels.get(self.response_key(meas, physics))
        return None if k is None else replace(k, m0=dict(physics.m0), seconds=0.0)

    def keep(self, kernels):
        """``{response_key: Kernels}`` computed in another process (the GPU worker) kept in this one's cache."""
        for key, k in kernels.items():
            self._store(key, k)

    def _store(self, key, k):
        self._kernels[key] = k
        while len(self._kernels) > 128:                                     # a bounded memory: the oldest goes first
            self._kernels.pop(next(iter(self._kernels)))

    def responses(self, entries):
        """What :meth:`prepare` costs for a run's ``entries`` ``[(meas, physics)]`` in a process forked from this one,
        in order: ``(state, n_meas, saves)`` for each entry not cached here, once per key; ``state`` is ``no_field``
        (the field tier off: nothing to compile), ``warm`` (a band compiled here, or by an earlier entry of the run)
        or ``cold`` (a band to compile first); ``saves`` the WM pack's saves the entry's classes span
        (:meth:`saves_spanned` of :meth:`windows_needed`). ``[budget]`` prices them (``response_<state>``)."""
        out = []; keys = set(); bands = set(self._compiled)
        for meas, ph in entries:
            key = self.response_key(meas, ph)
            if key in self._kernels or key in keys:
                continue
            keys.add(key)
            band = self._band(meas, ph)
            saves = self.saves_spanned(self.segments("wm", ph.wm_pack), self.windows_needed("wm", ph.wm_pack, meas))
            out.append(("no_field" if not ph.field else "warm" if band in bands else "cold", len(meas.bvals), saves))
            bands.add(band)
        return out

    def prepare(self, meas, physics=None):
        """The pose responses of a replay, on the CPU: per timing class of ``meas``, the WM and GM packs' pose
        responses at the run's tissue, field and specimen pose (dmipy-sim's closed form) contracted to the FOD's
        harmonics, free water's closed form; the true response curves for the explorer and the b = 0 weight of each
        tier alone (both at the first class's shells, within the windows ``meas`` reaches). Loads the windows of the
        chosen packs the classes reach (:meth:`windows_needed`, :meth:`pack`); computed every call and kept in this
        process's cache under :meth:`response_key` (:meth:`cached` reads it)."""
        from dmipy_sim.phantom import FreeWater
        from dmipy_sim.replay import so3
        t0 = time.perf_counter()
        wm_pack = self.pack("wm", physics.wm_pack, self.windows_needed("wm", physics.wm_pack, meas))
        gm_pack = self.pack("gm", physics.gm_pack, self.windows_needed("gm", physics.gm_pack, meas))
        for name in np.unique(meas.shape):                                  # the walks and the WM pack's tiers, now loaded
            self.validate(P.Protocol((P.Shell(str(name), 1.0, 1),), name="check"), physics)
        t_wm, t_gm, t_csf = physics.tissues(self.pools(wm_pack), self.pools(gm_pack))
        pose = specimen_pose(self.asset.R, physics.b0_direction)
        A = axis_map()
        n = len(meas.bvals)
        C_wm = np.zeros((n, N_COEF), np.complex128); e_gm = np.zeros(n, np.complex128); e_csf = np.zeros(n, np.complex128)
        classes = [np.flatnonzero(meas.shape == name) for name in np.unique(meas.shape)]
        seqs = [self.sequence(meas, rows, pose) for rows in classes]
        prof = self._profile_sequence(meas, classes[0], pose)             # the explorer's curves ride in the same pass
        batch = seqs + ([prof[0]] if prof is not None else [])
        r_wm = wm_pack.pose_responses(batch, tissue=t_wm, scanner=physics.scanner, pose=pose, keep=(LMAX, 0))
        r_gm = gm_pack.pose_responses(batch, tissue=t_gm, scanner=physics.scanner, pose=pose, keep=(0, 0))
        free = FreeWater(m0=1.0, tissue=t_csf)
        for rows, seq, rw, rg in zip(classes, seqs, r_wm, r_gm):
            C_wm[rows] = rw.retained(LMAX, 0) @ A
            e_gm[rows] = rg.retained(0, 0)[:, 0] * A[0, 0] * C00
            e_csf[rows] = free.response(seq)
        profile = None
        if prof is not None:
            seq, shells = prof
            axis = so3.Distribution.axis((0.0, 0.0, 1.0), lmax=LMAX, nmax=0)
            wm = np.abs(r_wm[-1].compose(axis)); gm = np.abs(r_gm[-1].retained(0, 0)[:, 0]); csf = np.abs(free.response(seq))
            k_ = len(PROFILE_ANGLES)
            profile = dict(b=shells, angles=PROFILE_ANGLES, wm=(wm[1:] / wm[0]).reshape(len(shells), k_),
                           gm=(gm[1:] / gm[0]).reshape(len(shells), k_)[:, 0], csf=(csf[1:] / csf[0]).reshape(len(shells), k_)[:, 0])
        b0 = self._tier_weights(meas, physics, wm_pack, gm_pack, pose, classes[0])
        floors = {"wm": _floor(wm_pack), "gm": _floor(gm_pack)}
        k = Kernels(C_wm, e_gm, e_csf, dict(physics.m0), floors, time.perf_counter() - t0, profile, b0)
        self._store(self.response_key(meas, physics), k)
        self._compiled.add(self._band(meas, physics))
        return k

    def _profile_sequence(self, meas, rows, pose):
        """``(seq, shells)``: the explorer's profile at the shells of ``meas``'s first class (``rows``): b = 0, then
        per shell a gradient at each of :data:`PROFILE_ANGLES` from the image's z, where the profile's fibre lies; None
        without a diffusion-weighted shell."""
        shells = np.unique(np.round(meas.bvals[rows][~meas.b0[rows]]))
        if not shells.size:
            return None
        th = np.radians(PROFILE_ANGLES)
        dirs = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], 1)
        b = np.concatenate([[0.0], np.repeat(shells, len(th))])
        g = np.concatenate([[[0.0, 0.0, 1.0]], np.tile(dirs, (len(shells), 1))])
        prof = P.Measurements(b, g, np.full(len(b), meas.shape[rows[0]]), np.zeros(len(b)), np.zeros(len(b)), np.zeros(len(b)))
        return self.sequence(prof, np.arange(len(b)), pose), shells

    def _tier_weights(self, meas, physics, wm_pack, gm_pack, pose, rows):
        """The b = 0 signal per tissue with each tier alone, over bare: ``relaxation`` (T2 at the echo, T1 over a
        stimulated echo's storage) per tissue, ``contact`` (the WM walls' survival) for WM; a tier that is off is
        absent. The b = 0 signal has no orientation, so one pose of the pack answers it."""
        from dmipy_sim.phantom import FreeWater
        seq = self._b0_sequence(str(meas.shape[rows[0]]), pose)
        out = {}
        alone = dict(relaxation=False, contact=False, field=False)
        if physics.relaxation:
            t_wm, t_gm, t_csf = replace(physics, **{**alone, "relaxation": True}).tissues(self.pools(wm_pack), self.pools(gm_pack))
            out["relaxation"] = {"wm": float(np.abs(wm_pack.replay(seq, tissue=t_wm))[0]), "gm": float(np.abs(gm_pack.replay(seq, tissue=t_gm))[0]),
                                 "csf": float(np.abs(FreeWater(m0=1.0, tissue=t_csf).response(seq))[0])}
        if physics.contact:
            t_wm, _, _ = replace(physics, **{**alone, "contact": True}).tissues(self.pools(wm_pack), self.pools(gm_pack))
            out["contact"] = {"wm": float(np.abs(wm_pack.replay(seq, tissue=t_wm))[0])}
        return out

    def replay(self, meas, physics=None, prepared=None):
        """The kernel route on the device: ``S = |F_wm C_wm^T + f_gm m0_gm e_gm + f_csf m0_csf e_csf|`` over the
        brain's voxels in float32 torch (the device's GPU when there is one), S0-normalised; ``s0_factor`` is every
        voxel's b = 0 signal in M0 units (M0 = 1: pure CSF before relaxation); the floor is zero (the packs' floors are
        the accuracy)."""
        import torch
        if prepared is None:
            raise ValueError("a brain replay needs its pose responses: Brain.prepare(meas, physics) first")
        t0 = time.perf_counter()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        k = prepared
        m0 = k.m0
        f = torch.as_tensor(self.f, device=dev)
        F = torch.as_tensor(self.fod_unit, device=dev) * (f[:, 0] * m0["wm"])[:, None]
        Cr = torch.as_tensor(k.wm.real.astype(np.float32), device=dev); Ci = torch.as_tensor(k.wm.imag.astype(np.float32), device=dev)
        g = (f[:, 1] * m0["gm"])[:, None]; c = (f[:, 2] * m0["csf"])[:, None]
        eg = torch.as_tensor(np.stack([k.gm.real, k.gm.imag]).astype(np.float32), device=dev)
        ec = torch.as_tensor(np.stack([k.csf.real, k.csf.imag]).astype(np.float32), device=dev)
        Sr = F @ Cr.T + g * eg[0][None, :] + c * ec[0][None, :]
        Si = F @ Ci.T + g * eg[1][None, :] + c * ec[1][None, :]
        S = torch.sqrt(Sr * Sr + Si * Si)
        S0 = S[:, torch.as_tensor(meas.b0, device=dev)].mean(1)
        live = S0 > 0
        dwi = torch.where(live[:, None], S / torch.where(live, S0, 1.0)[:, None], torch.nan)
        out = np.full(self.mask.shape + (len(meas.bvals),), np.nan, np.float32)
        out[self.mask] = dwi.cpu().numpy()
        factor = np.full(self.mask.shape, np.nan, np.float32)
        factor[self.mask] = torch.where(live, S0, torch.nan).cpu().numpy()
        return out, np.zeros(self.mask.shape, np.float32), factor, time.perf_counter() - t0

    def reconstruct(self, dwi, s0_factor, meas, mask):
        """The configuration's reconstruction (:func:`space.pipeline.reconstruction`) and the round trip against the
        asset: ``msmt``, three-tissue responses and multi-tissue CSD on the signal in M0 units, or ``tournier07``,
        the single-fibre response and single-tissue CSD on the S0-normalised signal; each step's time under a name
        that says which ran. ``extras``: the path, the recovered fractions (multi-tissue only), the estimated response
        curves at the explorer's profile, and the round-trip numbers (:meth:`roundtrip`)."""
        method = P.reconstruction(self.cfg)
        if method == "tournier07":
            sh, t, (S0, model) = P.csd(dwi, meas, mask, backend=self.backend)
            seconds = {f"csd · single-tissue: tournier07 response + csd_tournier07_{self.backend}": t}
            responses = {"wm": (S0, model)}; fractions = None
            path = f"single-tissue Tournier 2007 (DISCO_RECONSTRUCTION=tournier07): white_matter_response_tournier07 + csd_tournier07_{self.backend}"
        else:
            sh, fractions, (S0s, models), (t_resp, t_fit) = P.csd_msmt(dwi * s0_factor[..., None], meas, mask, backend=self.backend)
            seconds = {"csd · responses: three_tissue_response_dhollander16": t_resp, f"csd · MT-CSD: csd_msmt_{self.backend}": t_fit}
            responses = dict(zip(TISSUES, zip(S0s, models))); path = f"multi-tissue: three_tissue_response_dhollander16 + csd_msmt_{self.backend}"
        t0 = time.perf_counter()
        extras = dict(reconstruction=path, fractions=None if fractions is None else fractions.astype(np.float32),
                      responses=estimated_profile(responses, meas), roundtrip=self.roundtrip(sh, fractions))
        seconds["round trip · the FOD and fractions against the input"] = time.perf_counter() - t0
        return sh, seconds, extras

    def roundtrip(self, sh, fractions):
        """The recovered FOD against the asset's in the WM (the stop mask's voxels): the principal peaks' angle
        (median, 95 %), the Pearson of the l = 0 coefficient (the apparent fibre density), the share of voxels whose
        peak count (up to three) agrees; with fractions, their Pearson per tissue and median absolute difference
        over the brain."""
        wm = self.asset.stop
        rec = P.fod_peaks(np.asarray(sh)[wm]); ref = P.fod_peaks(self.asset.fod[wm])
        cosang = np.abs(np.einsum("ij,ij->i", rec[0][:, 0], ref[0][:, 0]))
        both = (rec[1][:, 0] > 0) & (ref[1][:, 0] > 0)
        ang = np.degrees(np.arccos(np.clip(cosang[both], 0, 1)))
        afd_rec = np.asarray(sh)[wm][:, 0] * (1.0 if fractions is None else fractions[wm][:, 0])   # the WM signal's l = 0 amplitude
        afd_in = self.asset.fod[wm][:, 0] * self.asset.fractions[wm][:, 0]
        out = dict(peak_angle_median=float(np.median(ang)) if ang.size else float("nan"), peak_angle_p95=float(np.quantile(ang, 0.95)) if ang.size else float("nan"),
                   afd_pearson=P.pearson(afd_rec, afd_in),
                   peak_count_agreement=float(np.mean(rec[2] == ref[2])), n_wm=int(wm.sum()),
                   peak_counts_recovered=np.bincount(rec[2], minlength=4)[:4].tolist(), peak_counts_input=np.bincount(ref[2], minlength=4)[:4].tolist())
        if fractions is not None:
            m = self.mask
            for k, t in enumerate(TISSUES):
                out[f"fraction_{t}_pearson"] = P.pearson(fractions[m][:, k], self.asset.fractions[m][:, k])
                out[f"fraction_{t}_mad"] = float(np.median(np.abs(fractions[m][:, k] - self.asset.fractions[m][:, k])))
        return out

    def tracking_inputs(self, sh, density):
        """``(field, seeds)``: the FOD turned from the image frame into the scanner frame on the asset's affine, the
        domain the white matter and the regions (leaving it ends a streamline), ``density^3`` seeds per WM voxel."""
        from dmipy_sim.replay.so3 import rotate_sh
        from dmipy_tract import FODField, seeds_from_mask
        a = self.asset
        world = np.zeros(np.shape(sh), np.float32)
        world[self.mask] = rotate_sh(np.asarray(sh)[self.mask], a.R).astype(np.float32)
        return FODField(world, a.affine, a.stop | (a.labels > 0)), seeds_from_mask(a.stop, a.affine, density=density)

    def reference(self, tracking, seeds):
        """The truth: the asset's own FOD tracked with ``tracking`` from ``seeds`` (the same tracker, settings, seeds
        and key as the run), its connectome and lobar sum; computed once per (settings, seeds) and reused (B shares
        A's)."""
        key = (tracking, len(seeds), float(np.asarray(seeds).sum()))
        if key in self._reference:
            return self._reference[key], 0.0
        t0 = time.perf_counter()
        fld, _ = self.tracking_inputs(self.asset.fod, tracking.density)
        tg, _ = P.track(fld, seeds, tracking, self.backend)
        M = self.regions.matrix(tg)
        ref = dict(matrix=M, lobar=self.regions.grouped(M), streamlines=len(tg))
        self._reference = {key: ref}
        return ref, time.perf_counter() - t0

    def score(self, M, reference):
        """Against the truth's connectome: Pearson of log(1 + count) over the region pairs (``pearson_log``), the
        pairs connected in the run only and in the truth only, and the same Pearson over the lobar groups' upper
        triangle with the diagonal (the within-group counts)."""
        T = reference["matrix"]; pairs = self.regions.pairs
        a, b = M[pairs], T[pairs]
        G, Gt = self.regions.grouped(M), reference["lobar"]
        gp = np.triu_indices(len(G))
        return dict(pearson_log=P.pearson(np.log1p(a), np.log1p(b)), connected_pairs=int((a > 0).sum()), truth_pairs=int((b > 0).sum()),
                    only_run=int(((a > 0) & (b == 0)).sum()), only_truth=int(((a == 0) & (b > 0)).sum()),
                    lobar_pearson_log=P.pearson(np.log1p(G[gp]), np.log1p(Gt[gp])), n_pairs=len(pairs[0]))

    def compare(self, a, b):
        """A against B over the region pairs: the Pearson of log(1 + count) between the two, the pairs connected in
        one only, and B's score minus A's."""
        pairs = self.regions.pairs
        ma, mb = a.matrix[pairs], b.matrix[pairs]
        return dict(pearson_ab=P.pearson(np.log1p(ma), np.log1p(mb)), only_a=int(((ma > 0) & (mb == 0)).sum()),
                    only_b=int(((mb > 0) & (ma == 0)).sum()), delta_log=b.score["pearson_log"] - a.score["pearson_log"])

    def score_key(self):
        return "pearson_log"

    def ladder_steps(self, physics):
        """The rungs below ``physics``: bare diffusion (the tiers off, the same proton densities and packs), then the
        tiers on one at a time."""
        if physics is None or physics.bare:
            return []
        steps = [("bare diffusion", replace(physics, **{q: False for q in self.TIERS}))]
        wanted = [q for q in self.TIERS if getattr(physics, q)]
        on = []
        for tier in wanted[:-1]:
            on.append(tier)
            steps.append(("+ " + " + ".join(on), replace(physics, **{q: q in on for q in self.TIERS})))
        return steps

    def ingredients(self, meas, physics=None, prepared=None):
        """The per-voxel maps of what the tissue multiplies into the b = 0 signal: ``m0`` (Σ f·M0), ``relaxation``
        (the voxel's b = 0 weight from T2 / T1 alone), ``contact`` (the share the WM walls' surface relaxivity leaves
        of it), and ``tiers``, the per-tissue weights they are made of."""
        if prepared is None:
            return None
        f = self.asset.fractions; m0 = prepared.m0
        w = {t: f[..., k] * m0[t] for k, t in enumerate(TISSUES)}
        total = sum(w.values())
        out = dict(m0=np.where(self.mask, total, np.nan).astype(np.float32), relaxation=None, contact=None, tiers=prepared.b0)
        with np.errstate(invalid="ignore", divide="ignore"):
            if "relaxation" in (prepared.b0 or {}):
                r = prepared.b0["relaxation"]
                relaxed = sum(w[t] * r[t] for t in TISSUES)
                out["relaxation"] = np.where(self.mask & (total > 0), relaxed / total, np.nan).astype(np.float32)
            if "contact" in (prepared.b0 or {}):
                c = prepared.b0["contact"]["wm"]
                out["contact"] = np.where(self.mask & (total > 0), (w["wm"] * c + w["gm"] + w["csf"]) / total, np.nan).astype(np.float32)
        return out

    def accuracy(self, res=None):
        a = self.asset.manifest
        rows = [["subject", str(a["subject"])], ["source", f"{a['source'].get('dataset')} ({a['source'].get('license')})"],
                ["asset", self.where], ["grid (cropped to the brain)", f"{self.mask.shape}, {int(self.mask.sum()):,} voxels, "
                                                                        f"{int(self.asset.stop.sum()):,} in the WM stop mask"],
                ["regions", f"{self.regions.n} ({len(self.regions.groups)} lobar groups)"],
                ["the scan's timing class", self.shapes[SCAN]["label"]],
                *[list(n) for n in self.asset.notes],
                ["replay", "exact composition of the packs' pose responses (dmipy-sim's phantom route), float32 on the device"],
                ["per-voxel split-half floor", "none: every voxel reads the same two walks; the packs' certified floors below are the accuracy"]]
        for (tissue, label), pk in self.packs.items():
            seg = self.segments(tissue, label); held = self.windows_held(pk, seg)
            rows.append([f"{tissue.upper()} pack {label}", f"{pk.meta.get('id')}, {pk.n_walkers:,} walkers, {self.walk_seconds(pk) * 1e3:g} ms walk, "
                                                          f"windows held / declared {held} / {seg['n']} of {float(seg['T']) * 1e3:g} ms "
                                                          f"({self.saves_spanned(seg, held):,} of {self.saves_spanned(seg, seg['n']):,} saves at "
                                                          f"{float(seg['T']) / (int(seg['n_t']) - 1) * 1e6:.4g} µs), certified floor (max) {_floor(pk):.4g}"])
        return rows + super().accuracy(res)

    def score_text(self, tag, res):
        s = res.score; physics = res.physics; rt = res.extras.get("roundtrip") or {}
        snr = P.b0_snr(res)
        noise = "" if snr is None else f", SNR {res.snr:g} at M0 = 1 ({snr['median']:.1f} at b = 0 in the median voxel)"
        return (f"**{tag}: connectome vs the input's, Pearson log(1 + count) {s['pearson_log']:.3f}** (lobar {s['lobar_pearson_log']:.3f}; "
                f"{s['only_run']} pairs in the run only, {s['only_truth']} in the input's only); FOD round trip: principal peak "
                f"{rt.get('peak_angle_median', float('nan')):.1f}° median / {rt.get('peak_angle_p95', float('nan')):.1f}° 95 %, AFD r "
                f"{rt.get('afd_pearson', float('nan')):.3f} ({res.protocol.n_meas} measurements, {physics.label() if physics else 'bare'}{noise}, "
                f"{len(res.tractogram):,} streamlines, {res.seconds['total']:.1f} s; {res.extras.get('reconstruction', '')})")

    def compare_text(self, c, knob):
        return (f"**A vs B: connectome Pearson log(1 + count) {c['pearson_ab']:.3f}**, {c['only_a']} pairs in A only, {c['only_b']} in B only, "
                f"B − A {c['delta_log']:+.3f} vs the input's connectome; B = A with {knob}.")

    def matrices(self, M, score, reference):
        from .. import viewers as V
        return V.pair_panel([(f"this run (Pearson log {score['pearson_log']:.3f})", M), ("the input FOD tracked the same way", reference["matrix"])],
                            f"{score['connected_pairs']} of {score['n_pairs']} pairs connected, the input's {score['truth_pairs']}; "
                            f"{score['only_run']} in the run only, {score['only_truth']} in the input's only", names=self.regions.names)

    def lobar(self, M, score, reference):
        from .. import viewers as V
        return V.pair_panel([(f"lobar groups, this run (Pearson log {score['lobar_pearson_log']:.3f})", self.regions.grouped(M)),
                             ("lobar groups, the input", reference["lobar"])], "the connectome summed over the lobar groups",
                            names=tuple(n for n, _ in self.regions.groups))

    @staticmethod
    def ingredient_layers(ing):
        if not ing:
            return [None, None, None]
        return [("proton density Σ f·M0 (relative to CSF)", ing["m0"], dict(vmin=0, vmax=1)),
                ("relaxation: the voxel's b = 0 weight from T2 / T1 at the echo", ing["relaxation"], dict(cmap="magma", vmin=0, vmax=1)) if ing.get("relaxation") is not None else None,
                ("contact: the share of the b = 0 signal the WM walls' ρ leaves", ing["contact"], dict(cmap="magma", vmax=1)) if ing.get("contact") is not None else None]

    @functools.cached_property
    def input_peaks(self):
        """The asset's WM FOD principal peaks and amplitudes (image frame), for the slices."""
        return P.peaks(self.asset.fod)

    def truth_view(self, dwi, meas, z, m):
        from .. import viewers as V
        pk, amp = self.input_peaks
        return V.dwi_slice(dwi, meas, z, m, peaks=pk, peak_amp=amp, overlay=True, label="the input FOD's peaks · ")

    def fractions_view(self, extras, z):
        from .. import viewers as V
        return V.fractions_pair(extras.get("fractions"), self.asset.fractions, z)

    def roundtrip_rows(self, res):
        rt = res.extras.get("roundtrip") or {}
        rows = [["reconstruction", res.extras.get("reconstruction", "")],
                ["principal peak vs the input's, WM voxels: median / 95 % (°)", f"{rt.get('peak_angle_median', float('nan')):.2f} / {rt.get('peak_angle_p95', float('nan')):.2f}"],
                ["AFD (the WM signal's l = 0 amplitude: the FOD's, times the WM fraction where one is fitted) Pearson, WM voxels", f"{rt.get('afd_pearson', float('nan')):.4f}"],
                ["peak count agreement (up to 3), WM voxels", f"{100 * rt.get('peak_count_agreement', float('nan')):.1f} % of {rt.get('n_wm', 0):,}"],
                ["voxels with 0 / 1 / 2 / 3 peaks: recovered", " / ".join(str(x) for x in rt.get("peak_counts_recovered", []))],
                ["voxels with 0 / 1 / 2 / 3 peaks: input", " / ".join(str(x) for x in rt.get("peak_counts_input", []))]]
        for t in TISSUES:
            if f"fraction_{t}_pearson" in rt:
                rows.append([f"{t.upper()} fraction vs the input's: Pearson / median |Δ|", f"{rt[f'fraction_{t}_pearson']:.4f} / {rt[f'fraction_{t}_mad']:.4f}"])
        if "fraction_wm_pearson" not in rt:
            rows.append(["fractions", "not estimated by the single-tissue reconstruction"])
        s = res.score
        rows += [["connectome vs the input's: Pearson log(1 + count), region pairs", f"{s['pearson_log']:.4f}"],
                 ["the same over the lobar groups", f"{s['lobar_pearson_log']:.4f}"],
                 ["pairs connected: run / input / run only / input only", f"{s['connected_pairs']} / {s['truth_pairs']} / {s['only_run']} / {s['only_truth']} of {s['n_pairs']}"],
                 ["streamlines: run / input", f"{len(res.tractogram):,} / {res.reference['streamlines']:,}"]]
        return rows

    def response_view(self, prepared, extras):
        from .. import viewers as V
        est = extras.get("responses")
        note = "" if est is not None and est.get("gm") is not None else "estimated: WM only, the single-tissue reconstruction"
        return V.response_curves(prepared.profile if prepared is not None else None, est, note=note)

    def extra_files(self, res, out_dir, stem):
        """The connectome and its lobar sum as CSV, the regions' names as the header."""
        import csv
        paths = []
        for what, M, names in (("connectome", res.matrix, self.regions.names), ("lobar", self.regions.grouped(res.matrix), tuple(n for n, _ in self.regions.groups))):
            path = os.path.join(out_dir, f"{stem}_{what}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f); w.writerow([""] + list(names))
                for name, row in zip(names, M):
                    w.writerow([name] + [f"{v:g}" for v in row])
            paths.append(path)
        return paths

    def truth_views(self):
        """The input tab: the region centroids in 3-D coloured by lobe, and a slice of the input (the WM FOD's principal
        peaks over the mean b = 0, the fractions, the labels)."""
        from .. import viewers as V
        a = self.asset
        pk, amp = P.peaks(a.fod)
        z = self.mask.shape[2] // 2
        return (V.regions3d(self.regions, self.mask.shape), V.input_slices(a.b0, pk, amp, a.fractions, a.labels, z))


def _warm_worker(cfg, where, entries, queue):
    """The warm-up's process (:meth:`Brain.warm_in_background`): its own :class:`Brain` on the same configuration
    and asset, every ``(meas, physics)`` of ``entries`` computed on the CPU and put on ``queue`` as
    ``(response_key, Kernels)``, then ``None``."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        src = Brain(cfg, asset=where)
        for meas, ph in entries:
            kernels = src.prepare(meas, ph)
            queue.put((src.response_key(meas, ph), kernels))
    finally:
        queue.put(None)


def _floor(pack):
    return float((pack.meta.get("fidelity") or {}).get("floor_max") or float("nan"))


@functools.lru_cache(maxsize=1)
def axis_map():
    """``(n_so3(8, 0), 45)``: a compact even-order density's harmonics to its SO(3) coefficients
    (:func:`dmipy_sim.replay.so3.axis_density_coeffs`, which is linear), so ``resp.retained(8, 0) @ axis_map()`` is the
    response contracted to the FOD's harmonics."""
    from dmipy_sim.replay import so3
    return np.stack([so3.axis_density_coeffs(e, LMAX, 0) for e in np.eye(N_COEF)], 1)


def estimated_profile(responses, meas):
    """The estimated response curves at the explorer's profile (the shells of ``meas``, :data:`PROFILE_ANGLES`): the
    WM response a fibre along z gives each gradient angle, GM's and CSF's isotropic signal, each over its b = 0
    value, from the models the reconstruction estimated (``{tissue: (S0, model)}``); ``None`` for a tissue it did not
    estimate."""
    from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
    rows = meas.shape == meas.shape[0]
    shells = np.unique(np.round(meas.bvals[rows & ~meas.b0]))
    if not shells.size:
        return None
    th = np.radians(PROFILE_ANGLES)
    dirs = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], 1)
    b = np.concatenate([[0.0], np.repeat(shells, len(th))]); g = np.concatenate([[[0.0, 0.0, 1.0]], np.tile(dirs, (len(shells), 1))])
    i = int(np.flatnonzero(rows)[0])
    sch = acquisition_scheme_from_bvalues(b * 1e6, g, delta=np.full(len(b), meas.delta[i]), Delta=np.full(len(b), meas.Delta[i]),
                                          TE=np.full(len(b), meas.TE[i]), b0_threshold=P.B0_MAX * 1e6)
    out = dict(b=shells, angles=PROFILE_ANGLES)
    for t in TISSUES:
        if t not in responses:
            out[t] = None
            continue
        _, model = responses[t]
        E = np.asarray(model(sch, mu=[0.0, 0.0]) if t == "wm" else model(sch), np.float64)
        E = (E[1:] / E[0]).reshape(len(shells), len(th))
        out[t] = E if t == "wm" else E[:, 0]
    return out
