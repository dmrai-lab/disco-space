"""The brain source's costs, measured: the packs' pose responses (JAX on the CPU, as in the page's process and the
GPU worker of a ZeroGPU Space) and every device stage of a run (the contraction, the noise, the reconstruction, the
tracking and the truth's tracking at density 1, 2 and 4, the scoring, the ladder), first call and steady, the
streamline counts, the truth connectome's reproducibility at one key and its spread over keys (the score's floor),
and the size of the payload that crosses from the GPU worker to the page. Writes a JSON the ``[budget]`` of
``space/brain.toml`` is refit from.

    DISCO_RECONSTRUCTION=tournier07 python tools/measure_brain.py --asset DIR --wm-pack WM.rpk --gm-pack GM.rpk [--out brain_costs.json]

With ``--saves WM1.rpk,WM2.rpk,...`` it measures only the responses' cost by the WM pack's saves spanned (the
``per_save`` of ``[budget]``'s ``response_*``): per WM pack, in a fresh interpreter (nothing compiled), ``prepare`` of
one protocol (``--saves-protocol``: a manifest.json whose protocol rows are played, else the asset's own; retimed to
``--saves-class``, the scan's own class by default) cold, warm and with the field tier off, beside the saves the class spans on that pack; and the
least-squares slope of the seconds over the saves per state.

    python tools/measure_brain.py --asset DIR --gm-pack GM.rpk --saves A.rpk,B.rpk [--saves-protocol manifest.json] [--saves-class scan]
"""
import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from space import pipeline as P            # noqa: E402
from space.sources import brain as B       # noqa: E402


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def five_shells():
    """MASiVar's shape: b = 1000 / 1500 / 2000 / 2500 / 3000 x 96 directions and 15 b = 0, on the scan's timing."""
    return P.Protocol(tuple(P.Shell(B.SCAN, b, 96) for b in (1000, 1500, 2000, 2500, 3000)), n_b0=15, name="five shells x 96")


def saves_one(asset, wm_pack, gm_pack, protocol_manifest, shape):
    """``prepare``'s seconds cold (the first call), warm (the same band at another tissue) and with the field tier off
    (each the faster of two calls), on one WM pack in this process, with the saves the protocol's class spans on it."""
    cfg = P.config(os.path.join(P.HERE, "brain.toml"))
    cfg["asset"] = {"local": asset}
    cfg["packs"] = {"wm": [{"label": "wm", "uri": wm_pack}], "gm": [{"label": "gm", "uri": gm_pack}]}
    src = B.Brain(cfg)
    panel = B.Brain.panel(cfg)
    v = {c.name: c.value for row in panel.rows for c in row if c.name in panel.fields}
    v.update(wm_pack="wm", gm_pack="gm")
    ph = B.Brain.physics_from(cfg, v)
    if protocol_manifest:
        with open(protocol_manifest) as f:
            prot = B.scan_protocol(json.load(f), name="the manifest's protocol")
    else:
        prot = B.Brain.protocol(cfg, B.Brain.presets(cfg)[0])
    prot = P.retime(prot, shape)
    meas = src.validate(prot, ph)
    seg = src.segments("wm", "wm")
    saves = B.Brain.saves_spanned(seg, src.windows_needed("wm", "wm", meas))
    seconds = {}
    other = lambda f: replace(ph, T2={**ph.T2, "gm": ph.T2["gm"] * f})          # the same band at another tissue
    for state, physics in (("cold", [ph]), ("warm", [other(1.0001), other(1.0002)]), ("no_field", [replace(ph, field=False), replace(other(1.0001), field=False)])):
        times = []
        for phys in physics:
            t = time.perf_counter(); src.prepare(meas, phys); times.append(time.perf_counter() - t)
        seconds[state] = min(times)
    pk = src.packs[("wm", "wm")]
    return dict(pack=wm_pack, id=pk.meta.get("id"), n_walkers=int(pk.n_walkers), K=int(pk.K), segments=seg, saves=saves,
                n_meas=int(len(meas.bvals)), shape=shape, seconds=seconds)


def measure_saves(a):
    """:func:`saves_one` per WM pack of ``--saves``, each in a fresh interpreter, and the slope of the seconds over the
    saves per state (least squares with an intercept)."""
    import multiprocessing as mp
    rows = []
    for pack in a.saves.split(","):
        with mp.get_context("spawn").Pool(1) as pool:
            row = pool.apply(saves_one, (a.asset, pack, a.gm_pack, a.saves_protocol, a.saves_class))
        rows.append(row)
        print(json.dumps(row), flush=True)
    x = np.array([r["saves"] for r in rows], float)
    fit = {}
    for state in ("cold", "warm", "no_field"):
        y = np.array([r["seconds"][state] for r in rows])
        slope, icpt = np.polyfit(x, y, 1) if len(rows) > 1 else (float("nan"), float("nan"))
        fit[state] = dict(per_save=float(slope), intercept=float(icpt))
    out = dict(cpu_threads=os.environ.get("OMP_NUM_THREADS"), rows=rows, fit=fit)
    print(json.dumps(fit), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--wm-pack")
    ap.add_argument("--gm-pack", required=True)
    ap.add_argument("--out", default="brain_costs.json")
    ap.add_argument("--densities", default="1,2,4")
    ap.add_argument("--saves", default=None, help="WM packs, comma-separated: measure the responses' cost by the saves spanned only")
    ap.add_argument("--saves-protocol", default=None, help="a manifest.json whose protocol the saves measurement plays")
    ap.add_argument("--saves-class", default=B.SCAN, help="the timing class the saves measurement plays it on")
    a = ap.parse_args()
    if a.saves:
        return measure_saves(a)
    if not a.wm_pack:
        ap.error("--wm-pack is required (or --saves)")
    cfg = P.config(os.path.join(P.HERE, "brain.toml"))
    cfg["asset"] = {"local": a.asset}
    cfg["packs"] = {"wm": [{"label": "wm", "uri": a.wm_pack}], "gm": [{"label": "gm", "uri": a.gm_pack}]}
    import torch
    out = dict(device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", reconstruction=P.reconstruction(cfg), backend=P.backend(cfg))
    t0 = time.perf_counter(); src = B.Brain(cfg); out["load_seconds"] = time.perf_counter() - t0
    out["voxels"] = int(src.mask.sum()); out["wm_voxels"] = int(src.asset.stop.sum()); out["grid"] = list(src.mask.shape)
    panel = B.Brain.panel(cfg)
    v = {c.name: c.value for row in panel.rows for c in row if c.name in panel.fields}
    v.update(wm_pack="wm", gm_pack="gm")
    ph = B.Brain.physics_from(cfg, v)
    protocols = {"scan": B.Brain.protocol(cfg, B.Brain.presets(cfg)[0]), "five shells x 96": five_shells()}
    host = {}
    for name, prot in protocols.items():
        meas = src.validate(prot, ph)
        rows = {}
        for label, phys in [("every tier (cold)", ph), ("every tier (warm)", replace(ph, T2={**ph.T2, "gm": ph.T2["gm"] * 1.0001}))] + \
                           [(f"rung: {l}", r) for l, r in src.ladder_steps(ph)]:
            t = time.perf_counter(); src.prepare(meas, phys); rows[label] = time.perf_counter() - t
        host[name] = dict(n_meas=prot.n_meas, seconds=rows)
        print(name, json.dumps(host[name]), flush=True)
    out["host_prepare"] = host
    device = {}
    for name, prot in protocols.items():
        meas = src.validate(prot, ph)
        k = src.cached(meas, ph)
        rec = dict(n_meas=prot.n_meas)
        for rep in ("first", "steady"):
            sync(); t = time.perf_counter(); dwi, floor, fac, _ = src.replay(meas, ph, k); sync(); rec[f"replay_{rep}"] = time.perf_counter() - t
        t = time.perf_counter(); noisy = P.add_noise(dwi, 30.0, fac, seed=0, backend="torch"); rec["noise"] = time.perf_counter() - t
        signal = np.isfinite(dwi[..., 0])
        for rep in ("first", "steady"):
            sync(); t = time.perf_counter(); sh, secs, extras = src.reconstruct(noisy, fac, meas, signal); sync(); rec[f"reconstruct_{rep}"] = time.perf_counter() - t
        rec["reconstruct_rows"] = secs
        rec["roundtrip"] = extras["roundtrip"]
        ladder = [src.cached(meas, r) for _, r in src.ladder_steps(ph)]
        sync(); t = time.perf_counter(); src.ladder(meas, ph, ladder); sync(); rec["ladder"] = time.perf_counter() - t
        t = time.perf_counter(); src.ingredients(meas, ph, k); rec["ingredients"] = time.perf_counter() - t
        tracks = {}
        for d in [int(x) for x in a.densities.split(",")]:
            tr = B.Brain.tracking(cfg, density=d, max_angle=cfg["tracking"]["max_angle"], step=cfg["tracking"]["step_mm"], key=0)
            fld, seeds = src.tracking_inputs(sh, d)
            row = dict(seeds=int(len(seeds)))
            for rep in ("first", "steady"):
                sync(); t = time.perf_counter(); tg, _ = P.track(fld, seeds, tr, src.backend); sync(); row[f"track_{rep}"] = time.perf_counter() - t
            row["streamlines"] = len(tg); row["points"] = int(tg.n_points.sum())
            src._reference.clear()
            sync(); t = time.perf_counter(); ref, _ = src.reference(tr, seeds); sync(); row["truth"] = time.perf_counter() - t
            t = time.perf_counter(); M = src.regions.matrix(tg); s = src.score(M, ref); row["score_seconds"] = time.perf_counter() - t
            row["score"] = s; row["truth_streamlines"] = ref["streamlines"]
            src._reference.clear(); again, _ = src.reference(tr, seeds)
            row["truth_bit_for_bit"] = bool(np.array_equal(again["matrix"], ref["matrix"]))
            src._reference.clear(); other, _ = src.reference(replace(tr, key=1), seeds)
            pairs = src.regions.pairs
            row["truth_key0_vs_key1_pearson_log"] = P.pearson(np.log1p(ref["matrix"][pairs]), np.log1p(other["matrix"][pairs]))
            src._reference.clear()
            tracks[d] = row
            print(name, d, json.dumps({k_: v_ for k_, v_ in row.items() if k_ != "score"}), json.dumps(s), flush=True)
        rec["tracking"] = tracks
        res = P.Result(prot, meas, noisy.astype(np.float32), floor, fac, sh, tg, seeds, M, s, clean=dwi.astype(np.float32), extras=extras)
        t = time.perf_counter(); blob = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL); rec["payload_mb"] = len(blob) / 1e6; rec["pickle_seconds"] = time.perf_counter() - t
        device[name] = rec
        print(name, json.dumps({k_: v_ for k_, v_ in rec.items() if k_ not in ("tracking", "roundtrip")}), flush=True)
    out["device"] = device
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
