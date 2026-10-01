"""The DiSCo source: the DiSCo phantom's 40^3 grid replayed from its stored walk, tracked from the sixteen regions
and scored against the dataset's strand-count and cross-sectional-area matrices (CC BY 4.0, ``data/SOURCE.md``).

Two sources replay the same walk. :class:`Layout` (demo mode) is dmipy-sim's shape-moment layout: the DiSCo replay
pack contracted once per stored pulse timing ("class"), so a protocol on those classes at any b-values, directions,
tissue and field is an elementwise kernel on device-resident rows, in seconds. :class:`Columns` (full mode) is the
columnar replay pack itself, read once per run for any pulse timing, in minutes. Both are a :class:`Disco`: the
page's texts, the tissue panel of the white-matter catalogue, the regions, the ground truth and the GPU budget.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .. import pipeline as P

N_REGIONS = 16
VOXEL_M = 25e-6                                          # DiSCo's voxel; the strands' coordinate unit
B0_MODES = {"along z (the strands' frame)": P.B0_ALONG_Z, "transverse (x): 90° from z, as in a biplanar magnet like the Swoop": P.B0_TRANSVERSE}
FREE_B0 = "free (polar and azimuth angles below)"
DISCO_364 = "DiSCo 364"


def disco_protocol(cfg):
    """DiSCo's own 364-measurement protocol (its gradient table, its per-shell timing classes from ``[disco]``) as
    a Protocol with every row's own direction and b-value: the acquisition the reference volumes were replayed with.
    Returns it and the table rows in the protocol's order (the reference volumes follow the table's)."""
    bv = np.loadtxt(os.path.join(P.DATA_DIR, "DiSCo_gradients.bvals")).ravel()
    dirs = np.loadtxt(os.path.join(P.DATA_DIR, "DiSCo_gradients_dipy.bvecs"))
    if dirs.shape[0] == 3:
        dirs = dirs.T
    shells = cfg["disco"]["shells"]                                     # [{b, shape}] in the table's row order
    centres = np.array([s["b"] for s in shells])
    which = np.argmin(np.abs(bv[:, None] - centres[None, :]), axis=1)
    return P.protocol_from_rows(bv, dirs, which, lambda k, rows: P.Shell(shells[k]["shape"], float(np.round(bv[rows].mean())), int(len(rows))),
                                name=DISCO_364)


def pools(cfg):
    """The seeded pools the panel offers T2 and T1 for."""
    return tuple(cfg["physics"]["pools"])


def physics_fields(cfg):
    """The panel's inputs, in the run signature's order."""
    q = pools(cfg)
    return ("on", "field_T", "b0_mode", "theta", "phi", *[f"T2_{p}" for p in q], *[f"T1_{p}" for p in q],
            "rho", "chi_iso", "chi_aniso", "relaxation", "contact", "field")


def tissue_numbers(cfg):
    """The panel's inputs the catalogue fills in: ms, ms, µm/s, ppm on the page."""
    q = pools(cfg)
    return (*[f"T2_{p}" for p in q], *[f"T1_{p}" for p in q], "rho", "chi_iso", "chi_aniso")


class Disco(P.Source):
    """What a DiSCo replay source has beside its replay data: the DiSCo mask, the sixteen regions, the two
    ground-truth matrices, the config's timing classes and seeded pools, the compute backend; the page's DiSCo texts,
    presets, tissue panel and knobs; the tracker seeded in the regions on the voxel grid (the millimetre frame is
    the voxel frame); the score against the strand count and the cross-sectional area."""
    TIERS = ("relaxation", "contact", "field")

    def __init__(self, cfg):
        import nibabel as nib
        super().__init__(cfg)
        self.pools = pools(cfg)
        self.mask = np.asarray(nib.load(os.path.join(P.DATA_DIR, "DiSCo_mask.nii.gz")).dataobj) > 0
        self.rois = np.asarray(nib.load(os.path.join(P.DATA_DIR, "DiSCo_ROIs.nii.gz")).dataobj).astype(np.int32)
        self.gt_count = np.loadtxt(os.path.join(P.DATA_DIR, "DiSCo_Connectivity_Matrix_Strands_Count.txt"))
        self.gt_area = np.loadtxt(os.path.join(P.DATA_DIR, "DiSCo_Connectivity_Matrix_Cross-Sectional_Area.txt"))
        self.affine = np.eye(4)
        self.regions = P.Regions(self.rois, self.affine, tuple(str(k) for k in range(1, N_REGIONS + 1)))

    def check_grid(self, grid_shape, what):
        if tuple(grid_shape) != self.mask.shape:
            raise ValueError(f"{what}'s grid {tuple(grid_shape)} is not the mask's {self.mask.shape}")

    # ---- the page's side ----
    @classmethod
    def describe(cls, cfg):
        full = P.mode(cfg) == "full"
        shapes = cfg["shapes"]
        return dict(
            title="DiSCo replay to tractogram",
            heading=("# DiSCo: one Monte-Carlo walk, any acquisition, a connectome\n"
                     "The DiSCo phantom's walkers were simulated once (SubstrateCommons/disco-replay). Choose an acquisition; the Space "
                     "replays the whole 40³ grid from the stored walk, adds noise, fits constrained spherical deconvolution, tracks from "
                     "the sixteen regions and scores the connectome against the ground truth. The progress bar names each stage."),
            acquisition=("**Full mode**: the columnar replay pack is read for every run, so a shell's δ / Δ / TE are free "
                         "(one TE and one pulse kind per run, square pulses, the field along z) and a Camino `.scheme` file "
                         "can be uploaded; a run takes minutes, the plan shown first says how many.") if full else
                        ("**Demo mode**: a shell's **pulse timing** (δ, Δ; TE 53.5 ms, square pulses) is one of the stored classes, "
                         "because the layout holds each walker's response to that pulse shape; the b-value, the directions and their "
                         "number, the tissue, the field, the SNR and the tracker are free. Stored: " + "; ".join(f"`{n}` = {shapes[n]['label']}" for n in shapes)
                         + ". The same image beside the columnar pack (`DISCO_MODE=full`) replays any timing, in minutes."),
            tissue=("The pack's walkers live in the intra- and extra-axonal pools (its spec names a myelin pool nobody was seeded "
                    "in). On this phantom the field's **direction** and the **stimulated echo** move the signal most; 3 T against "
                    "7 T on a PGSE is small (the 180° refocuses the static dephasing), and at 7 T the catalogue's two T2 coincide."),
            explorer=("What the replay made, before the noise and the tractography. **A** is the run you configured in the first tab "
                      "(its replay before the noise); **B** is the same run with the one knob you chose there changed, and exists only "
                      "when a knob is set. **Ingredients**: what each tier multiplies into every walker's term, reduced per voxel. "
                      "**Layers**: A's walk replayed with A's tiers switched on one at a time, ending at A itself, then B; look at a "
                      "layer, its difference to the previous one, or B minus A, in the DWI itself or in the tensor's MD and FA from the "
                      "b ≤ 1500 shells; divide by the replay floor to see where a difference means something. A null result is a "
                      "result: 7 T against 3 T at the catalogue's tissue moves the median voxel by 0.005 (99 % of voxels under 0.02), under "
                      "the replay floor of about 0.01: the catalogue gives both seeded pools the same T2 at 7 T (47 ms), so the "
                      "relaxation tier re-weights nothing between them, the contact tier does not depend on the field, and the sheath "
                      "field's dephasing, though it grows from 0.10 to 0.24 rad of spread, moves the magnitude by 0.004 in the median voxel."),
            results_tab="4 · DiSCo results",
            truth_tab="2 · ground truth",
            truth_labels=("the strands", "the ground-truth matrices"),
            stages={"replay": "replaying the grid from the stored walk", "noise": "adding Rician noise", "csd": "fitting CSD (order 8)",
                    "track": "tracking from the sixteen regions", "score": "scoring the connectome"},
            fixed=["the DiSCo phantom's geometry and its one stored walk", "the sixteen regions and the ground-truth matrices",
                   "the pulse timing classes of the layout (demo mode)"],
            accuracy=("A replay is a measured approximation of the stored walk, not a rendering: the walk's two halves are replayed "
                      "separately and their disagreement per voxel is the **floor** below which a signal difference means nothing; "
                      "the layout keeps K temporal bands of each walker's path and the **band error** is what the dropped bands "
                      "would have added at the built gradient amplitude."),
            ingredients=("relaxation tier: intra-axonal weight fraction", "contact tier: the walkers' wall contact", "field tier: dephasing phase spread at the echo"),
            truth="the dataset's strand-count and cross-sectional-area matrices: Pearson over the 120 region pairs",
            ladder=("Replay DWI Explorer: replay A's tier ladder too (bare, +relaxation, +contact; noise-free; about 30 s more of GPU time at DiSCo 364)"),
            views=dict(truth=False, fractions=False, lobar=False, roundtrip=False, response=False, floor=True),
            files="disco", step_unit="voxels; the grid is the mm frame",
            volumes_label="DWI (.nii.gz) with bvals/bvecs, and the FOD SH field (.nii.gz, tournier07 order 8)",
            run_label="replay → CSD → track → score")

    @classmethod
    def presets(cls, cfg):
        return [DISCO_364] + list(cfg["presets"])

    @classmethod
    def protocol(cls, cfg, name):
        if name == DISCO_364:
            return disco_protocol(cfg)[0]
        return P.preset_protocol(cfg, name)

    @classmethod
    def panel(cls, cfg):
        phys = cfg["physics"]; f0 = float(phys["default_field"]); c0 = cls.catalogue_numbers(cfg, f0); q = pools(cfg); n = len(q)
        C = P.Control
        rows = (
            (C("on", "checkbox", "evaluate the walk in tissue at a field (off: bare diffusion)", bool(phys["default_on"])),),
            (C("field_preset", "field_preset", "field preset", f"{f0:g} T", tuple(f"{float(f):g} T" for f in phys["fields"])),
             C("field_T", "slider", "B0 (T)", f0, minimum=0.05, maximum=12.0, step=0.001)),
            (C("b0_mode", "dropdown", "B0 direction", list(B0_MODES)[0], tuple(B0_MODES) + (FREE_B0,)),
             C("theta", "slider", "polar angle from z (°)", 0, minimum=0, maximum=180, step=1),
             C("phi", "slider", "azimuth from x (°)", 0, minimum=0, maximum=360, step=1)),
            (C("relaxation", "checkbox", "relaxation (T2, T1)", True), C("contact", "checkbox", "contact (surface relaxivity ρ)", True),
             C("field", "checkbox", "field (myelin susceptibility)", True)),
            tuple(C(f"T2_{p}", "number", f"T2 {p} (ms)", c0[k]) for k, p in enumerate(q)),
            tuple(C(f"T1_{p}", "number", f"T1 {p} (ms)", c0[n + k]) for k, p in enumerate(q)),
            (C("rho", "number", "ρ (µm/s)", c0[2 * n]), C("chi_iso", "number", "χ_iso of the sheath, the field source (ppm)", c0[2 * n + 1]),
             C("chi_aniso", "number", "Δχ_a of the sheath (ppm)", c0[2 * n + 2])),
            (C("catalogue_note", "catalogue_note", value=c0[-1]), C("reset", "reset", "reset to the catalogue at this field")),
        )
        return P.Panel(rows, physics_fields(cfg), tissue_numbers(cfg), tuple(float(f) for f in phys["fields"]))

    @classmethod
    def catalogue_numbers(cls, cfg, field_T):
        """The catalogue's white matter at ``field_T`` in the page's units (ms, µm/s, ppm), in :func:`tissue_numbers`
        order, plus the note that says which cited field it came from: what the reset button and the field presets
        fill in."""
        q = pools(cfg)
        c = P.catalogue(float(field_T), q)
        note = (f"catalogue values at {c['catalogue_field']:g} T" if abs(c["catalogue_field"] - float(field_T)) < 1e-9
                else f"the catalogue has no cited relaxation at {float(field_T):g} T: nearest is {c['catalogue_field']:g} T, edit as you see fit")
        return ([c["T2"][p] * 1e3 for p in q] + [c["T1"][p] * 1e3 for p in q]
                + [c["rho"] * 1e6, c["chi_iso"] * 1e6, c["chi_aniso"] * 1e6, note])

    @classmethod
    def physics_from(cls, cfg, values, gradient=None):
        """The panel (page units: ms, µm/s, ppm) as a :class:`space.pipeline.Physics` in SI, or None when it is off.
        The DiSCo source plays ideal pulses: a gradient class is refused by name (its slew is not applied here)."""
        if gradient is not None:
            raise ValueError(f"the DiSCo source plays ideal pulses; the scanner's gradient limit {gradient!r} only checks what it can play. "
                             "Choose the limit-free scanner, or the brain Space for finite slew")
        if not values["on"]:
            return None
        q = pools(cfg)
        u = B0_MODES[values["b0_mode"]] if values["b0_mode"] in B0_MODES else P.b0_direction(values["theta"], values["phi"])
        return P.Physics(field_T=float(values["field_T"]), b0_direction=u,
                         T2={p: float(values[f"T2_{p}"]) * 1e-3 for p in q}, T1={p: float(values[f"T1_{p}"]) * 1e-3 for p in q},
                         rho=float(values["rho"]) * 1e-6, chi_iso=float(values["chi_iso"]) * 1e-6, chi_aniso=float(values["chi_aniso"]) * 1e-6,
                         relaxation=bool(values["relaxation"]), contact=bool(values["contact"]), field=bool(values["field"]))

    @classmethod
    def knobs(cls, cfg):
        """The one-knob changes B can make to A: the field (with the catalogue's tissue at it), the field's direction,
        a tier off, the tissue off, the noise, or every shell's pulse timing; ``{label: (kind, value)}``."""
        out = {P.NO_KNOB: None}
        for f in cfg["physics"]["fields"]:
            out[f"field → {float(f):g} T (catalogue tissue at that field)"] = ("field", float(f))
        for label in B0_MODES:
            out[f"B0 direction → {label}"] = ("b0", label)
        for tier in cls.TIERS:
            out[f"{tier} tier → off"] = ("tier", tier)
        out["tissue → off (bare diffusion)"] = ("bare", None)
        for v in (10, 100):
            out[f"SNR → {v}"] = ("snr", float(v))
        out["noise → off"] = ("snr", None)
        for name, sh in cfg["shapes"].items():
            out[f"every shell's pulse timing → {name} ({sh['label']})"] = ("shape", name)
        return out

    @classmethod
    def apply_knob(cls, cfg, change, protocol, snr_on, snr, values):
        """B's settings: A's with ``change`` (a value of :meth:`knobs`) applied; ``(protocol, snr_on, snr, values)``.
        A knob that changes the tissue panel needs A's panel on, so that B differs from A in that one thing."""
        kind, value = change
        v = dict(values)
        if kind in ("field", "b0", "tier") and not v["on"]:
            raise ValueError(f"the knob {kind!r} changes the tissue panel, which is off for A: switch it on, or choose another knob")
        if kind == "field":
            v["field_T"] = value; v.update(zip(tissue_numbers(cfg), cls.catalogue_numbers(cfg, value)))
        elif kind == "b0":
            v["b0_mode"] = value
        elif kind == "tier":
            v[value] = False
        elif kind == "bare":
            v["on"] = False
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
        return dict(density=(1, 4, t["density"], 1, "seeds per region voxel (density³)"),
                    max_angle=(10, 60, t["max_angle"], 1, "max angle (°)"),
                    step=(0.25, 1.0, t["step_mm"], 0.05, "step (voxels; the grid is the mm frame)"))

    @classmethod
    def tracking(cls, cfg, *, density, max_angle, step, key):
        return P.Tracking(density=int(density), step_mm=float(step), max_angle=float(max_angle), max_steps=int(cfg["tracking"]["max_steps"]), key=int(key))

    @classmethod
    def estimated_seconds(cls, cfg, protocol, *, density, knob, n_keys, ladder, responses=()):
        """Measured on the pool at DiSCo 364 with every tier (#7, dmipy-sim#522/#523): the worker's start and the
        payload's handoff 8 s, noise to scoring 8 s, the first replay 0.045 s per measurement (the tiles uploaded
        inside the call), each further replay on the resident tiles 0.032 s per measurement (the ladder is three, B
        one plus its 8 s of stages, a new field direction included), 10 s per extra tracker key; times 1.3, within 30
        and 480 s. DiSCo prepares nothing, so ``responses`` is empty."""
        n = protocol.n_meas
        secs = 16 + 0.045 * n + (3 * 0.032 * n if ladder else 0) + ((8 + 0.032 * n) if knob != P.NO_KNOB else 0) + 10 * (int(n_keys) - 1)
        return int(min(480, max(30, 1.3 * secs)))

    # ---- the run's side ----
    def tracking_inputs(self, sh, density):
        """``(field, seeds)``: the FOD field on the mask and the ``density^3`` seeds per region voxel, built once for
        however many tracker keys run on them."""
        from dmipy_tract import FODField, seeds_from_mask
        return FODField(sh, self.affine, self.mask | (self.rois > 0)), seeds_from_mask(self.rois > 0, self.affine, density=density)

    def reference(self, tracking, seeds):
        """The ground-truth matrices, the same for every run."""
        return dict(count=self.gt_count, area=self.gt_area), 0.0

    def score(self, M, reference):
        """Pearson correlations over the 120 region pairs with the strand-count and area matrices, and the pair
        bookkeeping (connected, ground-truth, false, missed)."""
        return score(M, reference["count"], reference["area"])

    def compare(self, a, b):
        return compare(a, b)

    def truth_views(self):
        """The ground truth tab: a sample of the strands in 3-D with the region markers, and the two matrices."""
        from dmipy_sim.io.strands import read_tck, read_diameters
        from .. import viewers as V
        strands = read_tck(os.path.join(P.DATA_DIR, "DiSCo_Strands_Trajectories.tck"), coordinate_unit_m=VOXEL_M)
        diameters = read_diameters(os.path.join(P.DATA_DIR, "DiSCo_Strands_Diameters.txt"), diameter_unit_m=1e-3)
        return (V.strands3d([s / VOXEL_M for s in strands], diameters, V.region_markers(self.regions), self.mask.shape),
                V.ground_truth_matrix(self.gt_count, self.gt_area, P.connected_pairs(self.gt_count)))

    def score_text(self, tag, res):
        s = res.score; physics = res.physics
        snr = P.b0_snr(res)
        noise = "" if snr is None else f", SNR {res.snr:g} at M0 = {snr['median']:.1f} at b = 0 in the median voxel"
        return (f"**{tag}: Pearson vs strand count {s['pearson_count']:.3f}, vs area {s['pearson_area']:.3f}** "
                f"({res.protocol.n_meas} measurements, {physics.label() if physics else 'bare diffusion'}"
                f"{noise}, {len(res.tractogram):,} streamlines, {res.seconds['total']:.1f} s; "
                f"replay floor median {P.floor_stats(res)['median']:.4f})")

    def compare_text(self, c, knob):
        return (f"**A vs B: connectome Pearson {c['pearson_ab']:.3f}**, {c['only_a']} pairs in A only, {c['only_b']} in B only, "
                f"B − A {c['delta_count']:+.3f} vs count, {c['delta_area']:+.3f} vs area; B = A with {knob}.")

    def matrices(self, M, score, reference):
        """The connectome beside the strand-count ground truth."""
        from .. import viewers as V
        return V.matrices(M, score, reference["count"])

    def score_key(self):
        return "pearson_count"

    @staticmethod
    def ingredient_layers(ing):
        """The intra-axonal weight fraction, the walls' contact (the survival at the run's rho with the contact tier,
        else the boundary local time), the field's dephasing spread."""
        if not ing:
            return [None, None, None]
        pool = ("intra-axonal weight fraction (relaxation tier re-weights it)", ing["intra_fraction"], dict(vmin=0, vmax=1))
        if ing.get("contact_survival") is not None:
            contact = ("contact survival exp(−ρ ℓ / D) at the run's ρ (ℓ the walkers' wall contact)", ing["contact_survival"], dict(cmap="magma", vmax=1))
        elif ing.get("wall_contact_um") is not None:
            contact = ("walkers' wall contact ℓ (boundary local time, µm)", ing["wall_contact_um"], dict(cmap="magma", unit="µm"))
        else:
            contact = None
        fld = (("spread over the voxel's walkers of the sheath field's dephasing phase at the echo (rad)", ing["field_rad"], dict(cmap="inferno", unit="rad"))
               if ing.get("field_rad") is not None else None)
        return [pool, contact, fld]


PAIRS = np.triu_indices(N_REGIONS, 1)                # the 120 region pairs


def score(M, gt_count, gt_area):
    """Pearson correlations over the 120 region pairs with the strand-count and area matrices, and the pair
    bookkeeping (connected, ground-truth, false, missed)."""
    return dict(pearson_count=P.pearson(M[PAIRS], gt_count[PAIRS]), pearson_area=P.pearson(M[PAIRS], gt_area[PAIRS]),
                connected_pairs=P.connected_pairs(M), gt_pairs=P.connected_pairs(gt_count),
                false_pairs=int(((M[PAIRS] > 0) & (gt_count[PAIRS] == 0)).sum()), missed_pairs=int(((M[PAIRS] == 0) & (gt_count[PAIRS] > 0)).sum()))


def compare(a, b):
    """A against B over the 120 region pairs: the Pearson between the two streamline-count matrices, the pairs
    connected in one only, and B's score minus A's."""
    ma, mb = a.matrix[PAIRS], b.matrix[PAIRS]
    return dict(pearson_ab=P.pearson(ma, mb), only_a=int(((ma > 0) & (mb == 0)).sum()), only_b=int(((mb > 0) & (ma == 0)).sum()),
                delta_count=b.score["pearson_count"] - a.score["pearson_count"], delta_area=b.score["pearson_area"] - a.score["pearson_area"])


class Layout(Disco):
    """Demo mode: dmipy-sim's shape-moment layout (``dmipy_sim.replay.shape_moments``) at the config's dataset
    revision, or a local copy; every stored class's timing is the config's, and a tissue needs the layout's tiers."""
    mode = "demo"

    @classmethod
    def load(cls, cfg, *, local=None):
        """The layout at ``local``, ``DISCO_MOMENTS``, or the config's dataset revision on the Hub."""
        return cls(cfg, local=local or os.environ.get("DISCO_MOMENTS"))

    def __init__(self, cfg, *, local=None):
        from dmipy_sim.replay.shape_moments import ShapeMoments
        super().__init__(cfg)
        d = cfg["data"]
        uri = local or f"hf://{d['repo']}/{d['moments']}"
        self.moments = ShapeMoments.open(uri, revision=d.get("revision") or None) if uri.startswith("hf://") else ShapeMoments(uri)
        self.resident = P.resident(cfg)
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
        meas = P.measurements(protocol, self.shapes)
        if physics and not physics.bare and not self.tiers:
            raise ValueError("this layout holds bare diffusion only: switch the tissue panel off")
        if physics and set(physics.pools) != set(self.pools):
            raise ValueError(f"the tissue names the pools {physics.pools}; this layout's are {self.pools}")
        return meas

    def replay(self, meas, physics=None, prepared=None):
        t0 = time.perf_counter()
        S = np.full(self.mask.shape + (len(meas.bvals),), np.nan)
        floor = np.zeros(self.mask.shape)
        tissue, scanner = P.tissue_and_scanner(physics, self.unseeded)
        b0 = physics.b0_direction if physics else P.B0_ALONG_Z
        for name in np.unique(meas.shape):
            rows = meas.shape == name
            S[..., rows], f = self.moments.image(name, meas.bvals[rows] * 1e6, meas.dirs[rows], backend=self.backend,
                                                 resident=True, tissue=tissue, scanner=scanner, b0_direction=b0)
            floor = np.fmax(floor, f)                    # resident within a run: release() drops the device copies after it
        if self.M0 is None:                              # the bare b = 0 map: the walker weight of every voxel
            self.M0 = self.moments.image(meas.shape[0], np.zeros(1), np.array([P.B0_ALONG_Z]), backend=self.backend, resident=True)[0][..., 0]
        dwi, factor = P.s0_normalised(S, meas.b0, self.M0)
        return dwi, floor, factor, time.perf_counter() - t0

    def release(self):
        if not self.resident:
            self.moments.release()

    def ingredients(self, meas, physics=None, prepared=None):
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
        tissue, scanner = P.tissue_and_scanner(physics, self.unseeded)
        b0 = physics.b0_direction if physics else P.B0_ALONG_Z
        maps = self.moments.tier_maps(str(meas.shape[0]), tissue, scanner, backend=self.backend, resident=True, b0_direction=b0)
        contact = maps["contact"]
        return dict(D_walk=float(self.moments.manifest["tiers"]["D_walk"]), intra_fraction=maps["pool"]["intra"],
                    wall_contact_um=None if contact is None else -contact * 1e6,
                    contact_survival=maps["contact_weight"] if (physics and physics.contact) else None,
                    field_rad=maps["phase_std"] if (physics and physics.field) else None)

    def default_physics(self):
        """The tissue panel's default (``[physics]``), or None when it starts off or the layout has no tiers."""
        p = self.cfg["physics"]
        return P.Physics.at(float(p["default_field"]), pools=self.pools) if (p["default_on"] and self.tiers) else None

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
        self.replay(P.measurements(disco, self.shapes), physics)
        rest = [name for name in self.shapes if name not in {s.shape for s in disco.shells}]
        if rest:
            self.replay(P.measurements(P.Protocol(tuple(P.Shell(name, 1000.0, 1) for name in rest), n_b0=1, name="warm"), self.shapes), physics)

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


class Columns(Disco):
    """Full mode: the columnar replay pack (:class:`dmipy_sim.replay.columnar.ColumnarPack`) at
    :func:`space.pipeline.columns_uri`, read once per run: one ScannerSequence per run (one echo time, one pulse kind, square
    pulses), the tissue and field of the run, the field along the pack's z."""
    mode = "full"
    RATE = {True: 45e6, False: 500e6}                # bytes/s read from the Hub, from a local disk

    def __init__(self, cfg, *, uri=None):
        from dmipy_sim.replay.columnar import ColumnarPack
        super().__init__(cfg)
        self.uri = uri or P.columns_uri(cfg)
        self.pack = ColumnarPack(self.uri, workers=int(cfg["mode"]["workers"]))
        self.check_grid(self.pack.grid.shape, "the pack")
        self.remote = self.uri.startswith("hf://")

    def kinds(self, meas):
        """The pulse kinds (``pgse`` / ``pgste``) of the classes the rows play; a free timing is PGSE."""
        return sorted({self.shapes[n].get("kind", "pgse") if n in self.shapes else "pgse" for n in np.unique(meas.shape)})

    def validate(self, protocol, physics):
        meas = P.measurements(protocol, self.shapes)
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
        tissue, scanner = P.tissue_and_scanner(physics, self.unseeded)
        plan = self.pack.plan(self.sequence(meas), tissue=tissue, scanner=scanner, tol=P.BAND_TOL)
        plan["estimated_seconds"] = plan["bytes"] / self.RATE[self.remote]
        return plan

    def replay(self, meas, physics=None, prepared=None, *, progress=None):
        from dmipy_sim.replay.study import Study, Protocol as SProtocol, Acquisition
        t0 = time.perf_counter()
        tissue, scanner = P.tissue_and_scanner(physics, self.unseeded)
        study = Study(SProtocol([Acquisition(self.sequence(meas), name="run")]), tissues=[tissue], scanners=[scanner])
        S, floor, plan = self.pack.image_study(study, tol=P.BAND_TOL, chunk_rows=1_000_000, progress=progress)
        self.last_plan = plan
        dwi, factor = P.s0_normalised(S[0], meas.b0)      # the pack's bare weights are not read here: the factor is 1
        return dwi, floor[0], factor, time.perf_counter() - t0

    def accuracy(self, res=None):
        m = self.pack.meta
        rows = [["source pack", str(m.get("id"))], ["source pack's certified median floor", f"{self.pack.floor:.4g}"],
                ["bands kept", str(self.pack.K)], ["band tolerance (× the pack's floor)", f"{P.BAND_TOL:g}"],
                ["pulses", "square (slew rate ∞), one TE and one pulse kind per run"]]
        plan = getattr(self, "last_plan", None)
        if plan:
            rows.append(["last replay read", f"{plan.get('rows', 0):,} rows, {plan.get('bytes', 0) / 1e9:.1f} GB, bands {plan.get('K')}, modes {plan.get('modes')}"])
        return rows + super().accuracy(res)
