"""The live Space through its Gradio API: the configuration's first preset at SNR 30 with the tissue panel's defaults,
then a custom two-shell protocol on the configuration's first timing class; prints the headline, the timings and the
seconds the worker spent on the packs' responses the page had not cached and, with ``--stream``, every stage update as it
arrives; downloads the figures and files to ``$LIVE_OUT``. ``--config`` is the configuration the Space serves
(``DISCO_CONFIG``: ``config.toml`` for the DiSCo Spaces, ``brain.toml`` for ``rfick/brain-zero``): the inputs are
built from its source's panel, presets and knobs, not spelled here.

    python tools/live.py [space] [--config brain.toml] [--stream] [--preset NAME] [--no-ladder] [--knob CHOICE]
"""
import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("space", nargs="?", default="rfick/disco")
    ap.add_argument("--config", default=os.environ.get("DISCO_CONFIG", "config.toml"), help="the configuration in space/ the Space serves")
    ap.add_argument("--stream", action="store_true", help="print every stage update as it arrives")
    ap.add_argument("--preset", action="append", help="an acquisition to run (default: the first preset, then custom shells)")
    ap.add_argument("--no-ladder", action="store_true", help="skip the explorer's tier ladder (the shortest GPU reservation)")
    ap.add_argument("--knob", default=None, help="B: the same run with this knob changed (a choice of the page's dropdown)")
    a = ap.parse_args()
    os.environ["DISCO_CONFIG"] = a.config                     # before the page module reads its configuration
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gradio_client import Client
    from huggingface_hub import get_token
    from space import app as A, pipeline as P, sources

    cfg = P.config()
    S = sources.source_class(cfg)
    panel = S.panel(cfg)
    defaults = {c.name: c.value for row in panel.rows for c in row}
    tc = S.tracking_controls(cfg)
    shapes = list(S.shapes_of(cfg))
    physics = [defaults[name] for name in panel.fields]

    def inputs(preset, *, knob, n_keys=1, ladder=True):
        shells = [True, shapes[0], 1000, 30, 12.0, 24.0, 53.5, True, shapes[0], 3000, 45, 8.0, 20.0, 53.5,
                  False, shapes[0], 3000, 90, 17.0, 30.0, 53.5, False, shapes[0], 6000, 60, 17.0, 30.0, 53.5]
        return [preset, 2, True, 30, tc["density"][2], tc["max_angle"][2], tc["step"][2], 0, None, knob, A.NO_SCANNER, n_keys, ladder, *physics, *shells]

    c = Client(a.space, token=get_token(), verbose=False, download_files=os.environ.get("LIVE_OUT", "/tmp/disco-live"), httpx_kwargs={"timeout": 900})
    for preset in a.preset or (S.presets(cfg)[0], A.CUSTOM):
        t0 = time.perf_counter()
        args = inputs(preset, knob=a.knob or A.NO_KNOB, ladder=not a.no_ladder)
        print(f"{preset}: reserves {A.estimated_seconds(*args)} s", flush=True)
        job = c.submit(*args, api_name="/run_pipeline")
        seen = 0
        while a.stream and not job.done():
            outs = job.outputs()
            for o in outs[seen:]:
                print(f"  +{time.perf_counter() - t0:5.1f} s: {str(o[1])[:100]}", flush=True)
            seen = len(outs)
            time.sleep(1)
        out = job.result()
        print(f"{preset}: {time.perf_counter() - t0:.1f} s wall; result received at unix {time.time():.1f}", flush=True)
        named = dict(zip([n for n in A.OUTPUTS if n not in ("result", "b_row")], out))   # the API returns neither the gr.State nor the gr.Row
        print(named["headline"]); print(named["timings"])
        rows = named["timings"]["data"] if isinstance(named["timings"], dict) else named["timings"]
        responses = [r[1] for r in rows if r[0] == A.RESPONSES_ROW]
        print(f"the packs' responses on the worker: {responses[0] if responses else 'none (the source prepares nothing)'} s", flush=True)
        print("tck", named["tck"], "volumes", named["volumes"], flush=True)


if __name__ == "__main__":
    main()
