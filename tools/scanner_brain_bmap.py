"""The b each brain voxel is delivered on the scan's b = 1000 shell (96 directions, δ 25 / Δ 55 ms, TE 100 ms,
square pulses) by each machine, the head centre at isocentre: b = gamma^2 int (int eps g)^2 with the delivered
waveform eps(t) [s(t) q_v,i + g0_v] (bore.delivered_moments), whose three time integrals are taken once.

    ASSET=/asset/dir python tools/scanner_brain_bmap.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dmipy_sim as d
from dmipy_sim.constants import GAMMA
from dmipy_sim.phantom.bore import delivered_moments
from space import pipeline as P
from space.sources import brain as B
cfg = P.config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "space", "brain.toml")); cfg["asset"] = {"local": os.environ["ASSET"]}
cfg["packs"] = {"wm": [{"label": "w", "uri": "x"}], "gm": [{"label": "g", "uri": "y"}]}
br = B.Brain(cfg)
grid, idx = br.bore_grid
t = B.Brain.shape_table(cfg, os.environ["ASSET"])["scan"]
prot = B.scan_protocol(br.asset.manifest, name="scan"); meas = P.measurements(prot, br.shapes)
rows = np.flatnonzero(np.round(meas.bvals) == 1000)
u = meas.dirs[rows]                                     # image frame = the bore grid's frame
seq = d.pgse([[0.0, 0.0, 1.0]], t["delta"], t["Delta"], bvalues=[1e9], TE=t["TE"], n_t=2000, slew_rate=np.inf)
g = float(np.abs(np.asarray(seq.G)).max()); s = np.asarray(seq.G)[0, :, 2] / g
sgn = seq.rf.sign(np.arange(seq.n_t) * seq.dt)
Fs = np.cumsum(sgn * s) * seq.dt; F0 = np.cumsum(sgn) * seq.dt
I = {k: GAMMA ** 2 * np.sum(a * b_) * seq.dt for k, (a, b_) in dict(ss=(Fs, Fs), s0=(Fs, F0), oo=(F0, F0)).items()}
b_nom = g ** 2 * I["ss"]
out = {}
maps = {}
for key in ("hyperfine_swoop_64mT", "siemens_magnetom_prisma_3T", "siemens_magnetom_terra_7T"):
    L = P.limits(key)
    radius = L.b0_validity_radius or L.gnl_validity_radius or 1.0
    r = np.linalg.norm(grid.offset_m(idx), axis=1)
    inside = r <= radius
    q_all = np.full((len(idx), len(rows), 3), np.nan); g0_all = np.zeros((len(idx), 3))
    dm = delivered_moments(L, grid, s, np.full(len(rows), g), u, voxels=idx[inside], transmit=True)
    q_all[inside] = dm.q
    if dm.g0 is not None:
        g0_all[inside] = dm.g0
    bq = np.einsum("vmi,vmi->vm", q_all, q_all) * I["ss"] + 2 * np.einsum("vmi,vi->vm", q_all, g0_all) * I["s0"] + (g0_all ** 2).sum(1)[:, None] * I["oo"]
    frac = bq / b_nom
    mean = np.nanmean(frac, 1)
    vol = np.full(br.mask.shape, np.nan); vol[tuple(br.vox.T)] = mean
    maps[key] = vol
    kap = np.full(len(idx), np.nan)
    if dm.kappa is not None:
        kap[inside] = dm.kappa
    out[key] = dict(voxels=int(len(idx)), inside_anchor=int(inside.sum()), anchor_cm=radius * 100,
                    direction_mean_b_fraction=dict(median=float(np.nanmedian(mean)), p01=float(np.nanquantile(mean, 0.01)), p99=float(np.nanquantile(mean, 0.99)),
                                                   min=float(np.nanmin(mean)), max=float(np.nanmax(mean))),
                    per_direction_b_fraction=dict(p01=float(np.nanquantile(frac, 0.01)), p99=float(np.nanquantile(frac, 0.99)),
                                                  min=float(np.nanmin(frac)), max=float(np.nanmax(frac))),
                    direction_spread_median=float(np.nanmedian(np.nanmax(frac, 1) - np.nanmin(frac, 1))),
                    voxels_off_by_more_than_5pc=float(np.nanmean(np.abs(mean - 1) > 0.05)),
                    transmit_scale=None if dm.kappa is None else dict(median=float(np.nanmedian(kap)), min=float(np.nanmin(kap)), max=float(np.nanmax(kap))),
                    G_needed_mT_m=g * 1e3, G_max_mT_m=float(L.G_max) * 1e3)
    print(key, json.dumps(out[key], indent=1), flush=True)
np.savez_compressed("brain_bmap.npz", **{k: v for k, v in maps.items()}, affine=br.affine)
json.dump(out, open("brain_bmap.json", "w"), indent=1)
