"""The live Space through its Gradio API: the DiSCo 364 preset at SNR 30 with the default tissue panel at 3 T, then
a custom two-shell protocol; prints the headline and timings and, with ``--stream``, every stage update as it
arrives; downloads the figures and files to ``$LIVE_OUT``.

    python tools/live.py [space] [--stream] [--preset NAME]
"""
import argparse
import os
import sys
import time

from gradio_client import Client
from huggingface_hub import get_token

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from space import app as A          # noqa: E402  (the page's constants and the catalogue: the inputs are built, not spelled)


def inputs(preset, *, knob=A.NO_KNOB, n_keys=1, field=3.0):
    physics = [True, field, next(iter(A.B0_MODES)), 0, 0] + A.catalogue_numbers(field)[:-1] + [True, True, True]
    shells = [True, "d12-D24", 1000, 30, 12.0, 24.0, 53.5, True, "d8-D20", 3000, 45, 8.0, 20.0, 53.5,
              False, "d17-D30", 3000, 90, 17.0, 30.0, 53.5, False, "d17-D30", 6000, 60, 17.0, 30.0, 53.5]
    return [preset, 2, True, 30, 4, 30.0, 0.5, 0, None, knob, A.NO_SCANNER, n_keys, True, *physics, *shells]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("space", nargs="?", default="rfick/disco")
    ap.add_argument("--stream", action="store_true", help="print every stage update as it arrives")
    ap.add_argument("--preset", action="append", help="an acquisition to run (default: DiSCo 364, then custom shells)")
    a = ap.parse_args()
    c = Client(a.space, token=get_token(), verbose=False, download_files=os.environ.get("LIVE_OUT", "/tmp/disco-live"), httpx_kwargs={"timeout": 900})
    for preset in a.preset or ("DiSCo 364", A.CUSTOM):
        t0 = time.perf_counter()
        job = c.submit(*inputs(preset), api_name="/run_pipeline")
        seen = 0
        while a.stream and not job.done():
            outs = job.outputs()
            for o in outs[seen:]:
                print(f"  +{time.perf_counter() - t0:5.1f} s: {str(o[1])[:100]}", flush=True)
            seen = len(outs)
            time.sleep(1)
        out = job.result()
        print(f"{preset}: {time.perf_counter() - t0:.1f} s wall", flush=True)
        named = dict(zip([n for n in A.OUTPUTS if n not in ("result", "b_row")], out))   # the API returns neither the gr.State nor the gr.Row
        print(named["headline"]); print(named["timings"])
        print("tck", named["tck"], "volumes", named["volumes"], flush=True)


if __name__ == "__main__":
    main()
