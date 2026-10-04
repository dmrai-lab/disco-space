"""Effect sizes on the DiSCo page: each machine at 7.9 cm from isocentre (along R-L) against the same run on the
ideal scanner at the machine's field and direction (so only the delivered gradient and the transmit scale differ),
noiseless, the layout replayed on the CPU. Per machine: a protocol it can play on the layout's TE 53.5 ms classes.

    LAYOUT=/stamped/layout python tools/scanner_disco_effects.py   # [DIST_CM=7.9 AXIS='R-L (x)' DENSITY=4 ONLY=swoop OUT=...]
"""
import json, os, sys, time
from dataclasses import replace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from space import pipeline as P
from space.sources import disco as D
LAYOUT = os.environ["LAYOUT"]
cfg = P.config()
keep = ["d12-D24", "d17-D30", "d17.7-D35.8", "d10.2-D16.7"]
cfg["shapes"] = {k: cfg["shapes"][k] for k in keep}
src = D.Layout(cfg, local=LAYOUT)
src.backend = "jax"
axis = np.array(D.OFFSET_AXES[os.environ.get("AXIS", "R-L (x)")])
dist = float(os.environ.get("DIST_CM", "7.9")) * 1e-2
tracking = D.Disco.tracking(cfg, density=int(os.environ.get("DENSITY", "4")), max_angle=cfg["tracking"]["max_angle"], step=cfg["tracking"]["step_mm"], key=0)
runs = [("hyperfine_swoop_64mT", P.Protocol((P.Shell("d17.7-D35.8", 350.0, 60),), n_b0=3, name="Swoop-playable: b 350 x 60 on d17.7-D35.8")),
        ("siemens_magnetom_prisma_3T", D.Disco.protocol(cfg, "research 3-shell x 90")),
        ("siemens_magnetom_terra_7T", D.Disco.protocol(cfg, "research 3-shell x 90"))]
only = os.environ.get("ONLY")
out = {}
for key, prot in runs:
    if only and only not in key:
        continue
    L = P.limits(key)
    assert all(r[-1] for r in P.playable(prot, cfg["shapes"], key)), (key, P.playable(prot, cfg["shapes"], key))
    A = P.Physics.at(float(L.field_T), pools=D.pools(cfg), b0_direction=tuple(L.b0_axis or (0, 0, 1)))
    B = replace(A, scanner=key, offset_m=tuple(dist * axis))
    meas = src.validate(prot, B)
    t0 = time.perf_counter(); ra = P.run(src, prot, tracking=tracking, physics=A); ta = time.perf_counter() - t0
    t0 = time.perf_counter(); rb = P.run(src, prot, tracking=tracking, physics=B); tb = time.perf_counter() - t0
    m = np.isfinite(ra.dwi[..., 0]) & src.mask
    shells = {}
    for b in sorted(set(np.round(meas.bvals[~meas.b0]))):
        rows = np.flatnonzero(np.round(meas.bvals) == b)
        a = ra.dwi[m][:, rows].mean(1); bb = rb.dwi[m][:, rows].mean(1)
        rel = (bb - a) / a
        shells[str(b)] = dict(mean_A=float(np.mean(a)), mean_B=float(np.mean(bb)), rel_median=float(np.median(rel)), rel_p01=float(np.quantile(rel, 0.01)),
                              rel_p99=float(np.quantile(rel, 0.99)), abs_dwi_median=float(np.median(np.abs(rb.dwi[m][:, rows] - ra.dwi[m][:, rows]))),
                              abs_dwi_p99=float(np.quantile(np.abs(rb.dwi[m][:, rows] - ra.dwi[m][:, rows]), 0.99)))
    s0 = rb.s0_factor[m] / ra.s0_factor[m]
    out[key] = dict(protocol=prot.name, n_meas=int(len(meas.bvals)), offset_m=list(B.offset_m), field_T=float(L.field_T),
                    pearson_count=dict(ideal=ra.score["pearson_count"], machine=rb.score["pearson_count"]),
                    pearson_area=dict(ideal=ra.score["pearson_area"], machine=rb.score["pearson_area"]),
                    streamlines=dict(ideal=len(ra.tractogram), machine=len(rb.tractogram)), shells=shells,
                    s0_ratio_median=float(np.median(s0)), floor_median=float(np.nanmedian(rb.floor[m])),
                    seconds=dict(ideal=ta, machine=tb, replay_ideal=ra.seconds["replay"], replay_machine=rb.seconds["replay"]))
    print(key, json.dumps(out[key], indent=1), flush=True)
json.dump(out, open(os.environ.get("OUT", "disco_effects.json"), "w"), indent=1)
