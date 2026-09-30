"""The page's views: a DWI slice with the FODs' principal directions drawn over it, the replay floor slice, the
tractogram and the ground-truth strands as rotatable 3-D figures, the connectome beside the ground truth, the
spread over repeated runs, the timings. Matplotlib for the slices and matrices (PIL images), Plotly for the 3-D
views (figures the page rotates). Every number drawn comes in from :mod:`space.pipeline` or the page."""
from __future__ import annotations

import io

import numpy as np

from . import pipeline as P

REGION_COLOURS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
                  "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3"]
AXIS_COLOURS = ("#d62728", "#2ca02c", "#1f77b4")     # paths that mostly follow x, y, z


def _image(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _slice_axes(title):
    """One square axial-slice figure without ticks; ``(fig, ax)``."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


def _pair_panel(panels, suptitle, *, vmin=1.0):
    """Two 16 x 16 matrices side by side on a log colour scale (zeros blank): ``panels`` is ``[(title, M), ...]``."""
    plt = _mpl()
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8))
    ticks = range(0, P.N_REGIONS, 3)
    for ax, (title, A) in zip(np.atleast_1d(axes), panels):
        A = np.asarray(A, np.float64)
        top = max(float(A.max()), vmin * 2)
        im = ax.imshow(np.where(A > 0, A, np.nan), norm=LogNorm(vmin=vmin, vmax=top), cmap="viridis")
        ax.set_title(title, fontsize=9); ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([t + 1 for t in ticks]); ax.set_yticklabels([t + 1 for t in ticks])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(suptitle, fontsize=9)
    return _image(fig)


def dwi_slice(dwi, meas, z, m, *, peaks=None, peak_amp=None, overlay=False, label=""):
    """Axial slice ``z`` of measurement ``m`` (S0-normalised, grey 0..1); with ``overlay`` the principal FOD direction
    of every voxel as a short line coloured by |direction| (x, y, z -> r, g, b), scaled by its amplitude."""
    b = meas.bvals[m]
    fig, ax = _slice_axes(f"{label}slice z = {z}, measurement {m}: b = {b:g} s/mm², {meas.shape[m]}" + (", FOD peaks" if overlay else ""))
    sl = np.nan_to_num(dwi[:, :, z, m])
    ax.imshow(sl.T, origin="lower", cmap="gray", vmin=0, vmax=1, extent=(-0.5, dwi.shape[0] - 0.5, -0.5, dwi.shape[1] - 0.5))
    if overlay and peaks is not None:
        d = peaks[:, :, z]; a = peak_amp[:, :, z]
        ii, jj = np.nonzero(a > 0)
        if len(ii):
            L = 0.45 * a[ii, jj] / a[ii, jj].max()
            x0 = ii - L * d[ii, jj, 0]; x1 = ii + L * d[ii, jj, 0]; y0 = jj - L * d[ii, jj, 1]; y1 = jj + L * d[ii, jj, 1]
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection(np.stack([np.stack([x0, y0], 1), np.stack([x1, y1], 1)], 1),
                                             colors=np.abs(d[ii, jj]), linewidths=1.2))
    return _image(fig)


def floor_slice(floor, z, median, *, label=""):
    """Axial slice ``z`` of the replay's split-half floor per voxel (the disagreement of the two walker halves,
    the largest over the classes played), ``median`` the run's median floor for the title."""
    fig, ax = _slice_axes(f"{label}split-half floor, slice z = {z} (median {median:.4f})")
    sl = floor[:, :, z]
    im = ax.imshow(np.where(sl > 0, sl, np.nan).T, origin="lower", cmap="magma", vmin=0, vmax=max(float(np.nanmax(floor)), 1e-6))
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _image(fig)


def _lines3d(paths, colours, *, name, width=2.0, opacity=0.7):
    """One Plotly trace per colour group: the group's points concatenated with a NaN row between paths (a gap)."""
    import plotly.graph_objects as go
    traces = []
    for colour, group in zip(colours, paths):
        if not group:
            continue
        pts = np.concatenate(group).astype(np.float64)
        ends = np.cumsum([len(p) for p in group])
        pts = np.insert(pts, ends, np.nan, axis=0)
        traces.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines", line=dict(color=colour, width=width),
                                   opacity=opacity, name=name, showlegend=False, hoverinfo="skip", connectgaps=False))
    return traces


def _by_dominant_axis(paths):
    """Paths split by the axis their end-to-end vector mostly follows (x, y, z): the colour groups."""
    groups = [[], [], []]
    for pts in paths:
        v = np.abs(pts[-1] - pts[0])
        groups[int(np.argmax(v)) if v.max() > 0 else 0].append(pts)
    return groups


def region_markers(rois):
    """The region centroids as labelled Plotly markers (built once per source; the 3-D views take it)."""
    import plotly.graph_objects as go
    cs = []
    for k in range(1, P.N_REGIONS + 1):
        ijk = np.argwhere(rois == k)
        if len(ijk):
            cs.append((k, ijk.mean(0)))
    return go.Scatter3d(x=[c[1][0] for c in cs], y=[c[1][1] for c in cs], z=[c[1][2] for c in cs], mode="markers+text",
                        text=[str(c[0]) for c in cs], textposition="top center",
                        marker=dict(size=5, color=[REGION_COLOURS[c[0] - 1] for c in cs]), name="regions", showlegend=False)


def _figure3d(paths, regions, shape, title, *, name, width, opacity):
    """Paths (voxel coordinates) coloured by dominant axis with the region markers, in a cube of ``shape``."""
    import plotly.graph_objects as go
    fig = go.Figure(data=_lines3d(_by_dominant_axis(paths), AXIS_COLOURS, name=name, width=width, opacity=opacity) + [regions])
    fig.update_layout(scene=dict(xaxis=dict(range=[0, shape[0]], title="x (voxels)"), yaxis=dict(range=[0, shape[1]], title="y"),
                                 zaxis=dict(range=[0, shape[2]], title="z"), aspectmode="cube"),
                      margin=dict(l=0, r=0, t=30, b=0), height=560, title=dict(text=title, font=dict(size=13)))
    return fig


def _sample(items, n, seed):
    pick = np.random.default_rng(seed).choice(len(items), min(n, len(items)), replace=False)
    return [np.asarray(items[i]) for i in pick]


def tractogram3d(tg, regions, shape, *, n=1500, seed=0, total=None):
    """A random sample of ``n`` streamlines in 3-D, coloured by dominant axis (x red, y green, z blue), the region
    markers; ``total`` is the run's streamline count when ``tg`` is already a sample."""
    paths = [p for p in _sample(tg, n, seed) if len(p) > 1]
    return _figure3d(paths, regions, shape, f"{len(paths):,} of {total or len(tg):,} streamlines (drag to rotate)", name="streamlines", width=2.0, opacity=0.7)


def strands3d(strands, diameters_m, regions, shape, *, n=800, seed=0):
    """A random sample of ``n`` ground-truth strands (centerlines in voxel units, one line width) coloured by
    dominant axis, the region markers; the diameters' range (metres in, µm in the title)."""
    paths = _sample(strands, n, seed)
    return _figure3d(paths, regions, shape, f"{len(paths):,} of {len(strands):,} ground-truth strands, diameters "
                     f"{1e6 * diameters_m.min():.1f}-{1e6 * diameters_m.max():.1f} µm (drag to rotate)", name="strands", width=1.5, opacity=0.6)


def matrices(M, score, gt_count):
    """Ours beside the strand-count ground truth on a log colour scale, with the score's numbers."""
    return _pair_panel([(f"streamline counts (Pearson {score['pearson_count']:.3f} vs count, {score['pearson_area']:.3f} vs area)", M),
                        ("ground truth: strand count", gt_count)],
                       f"{score['connected_pairs']} of {len(P.PAIRS[0])} pairs connected, ground truth {score['gt_pairs']}; "
                       f"{score['false_pairs']} false, {score['missed_pairs']} missed")


def ground_truth_matrix(gt_count, gt_area, n_pairs):
    """The two ground-truth matrices; ``n_pairs`` is how many of the 120 pairs the strands connect."""
    return _pair_panel([("strand count", gt_count), ("cross-sectional area", gt_area)],
                       f"the ground truth: {n_pairs} connected pairs of {len(P.PAIRS[0])}", vmin=max(float(gt_area[gt_area > 0].min()), 1e-3))


def spread_matrices(spread):
    """The mean streamline count per pair beside its standard deviation over the repeated runs."""
    return _pair_panel([(f"mean count over {spread['n']} keys", spread["mean"]), ("standard deviation over the keys", spread["std"])],
                       f"Pearson vs count {spread['pearson_mean']:.3f} ± {spread['pearson_std']:.3f}; median pair CV {spread['cv_median']:.2f}; "
                       f"{spread['pairs_always']} pairs in every run, {spread['pairs_any']} in any")


def timings_rows(seconds, load_seconds):
    """The stage times as table rows, the source's load time (once per process) first."""
    rows = [["source loaded and warmed (once per process)", f"{load_seconds:.1f}"]]
    return rows + [[k, f"{v:.2f}"] for k, v in seconds.items()]
