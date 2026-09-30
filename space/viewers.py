"""The page's views: a DWI slice with the FODs' principal directions drawn over it, the tractogram and the ground-truth
strands as rotatable 3-D figures, the connectome beside the ground truth, the timings. Matplotlib for the slices and
matrices (PIL images), Plotly for the 3-D views (figures the page rotates). Every number comes from
:mod:`space.pipeline`; nothing is derived here."""
from __future__ import annotations

import io

import numpy as np

from . import pipeline as P

REGION_COLOURS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
                  "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3"]


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


def dwi_slice(dwi, meas, z, m, *, peaks=None, peak_amp=None, overlay=False, label=""):
    """Axial slice ``z`` of measurement ``m`` (S0-normalised, grey 0..1); with ``overlay`` the principal FOD direction
    of every voxel as a short line coloured by |direction| (x, y, z -> r, g, b), scaled by its amplitude."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
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
    b = meas.bvals[m]
    ax.set_title(f"{label}slice z = {z}, measurement {m}: b = {b:g} s/mm², {meas.shape[m]}" + (", FOD peaks" if overlay else ""), fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return _image(fig)


def _lines3d(paths, colours, *, name, width=2.0, opacity=0.7):
    """One Plotly trace per colour group: the paths' points joined, ``None`` between paths."""
    import plotly.graph_objects as go
    traces = []
    for colour, group in zip(colours, paths):
        if not group:
            continue
        xs, ys, zs = [], [], []
        for pts in group:
            xs += list(pts[:, 0]) + [None]; ys += list(pts[:, 1]) + [None]; zs += list(pts[:, 2]) + [None]
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=colour, width=width), opacity=opacity,
                                   name=name, showlegend=False, hoverinfo="skip"))
    return traces


def _by_dominant_axis(paths):
    """Paths split by the axis their end-to-end vector mostly follows (x, y, z): the colour groups."""
    groups = [[], [], []]
    for pts in paths:
        v = np.abs(pts[-1] - pts[0])
        groups[int(np.argmax(v)) if v.max() > 0 else 0].append(pts)
    return groups


def _scene(shape):
    return dict(xaxis=dict(range=[0, shape[0]], title="x (voxels)"), yaxis=dict(range=[0, shape[1]], title="y"),
                zaxis=dict(range=[0, shape[2]], title="z"), aspectmode="cube")


def _regions(rois):
    """Region centroids as labelled markers."""
    import plotly.graph_objects as go
    cs = []
    for k in range(1, P.N_REGIONS + 1):
        ijk = np.argwhere(rois == k)
        if len(ijk):
            cs.append((k, ijk.mean(0)))
    return go.Scatter3d(x=[c[1][0] for c in cs], y=[c[1][1] for c in cs], z=[c[1][2] for c in cs], mode="markers+text",
                        text=[str(c[0]) for c in cs], textposition="top center",
                        marker=dict(size=5, color=[REGION_COLOURS[c[0] - 1] for c in cs]), name="regions", showlegend=False)


def tractogram3d(tg, rois, shape, *, n=1500, seed=0):
    """A random sample of ``n`` streamlines in 3-D, coloured by dominant axis (x red, y green, z blue), the region
    centroids labelled."""
    import plotly.graph_objects as go
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tg), min(n, len(tg)), replace=False)
    paths = [np.asarray(tg[i]) for i in pick if len(tg[i]) > 1]
    fig = go.Figure(data=_lines3d(_by_dominant_axis(paths), ("#d62728", "#2ca02c", "#1f77b4"), name="streamlines") + [_regions(rois)])
    fig.update_layout(scene=_scene(shape), margin=dict(l=0, r=0, t=30, b=0), height=560,
                      title=dict(text=f"{len(paths):,} of {len(tg):,} streamlines (drag to rotate)", font=dict(size=13)))
    return fig


def strands3d(strands, diameters, rois, shape, *, n=800, seed=0):
    """A random sample of the ground-truth strands (centerlines in voxel units), line width by diameter class,
    coloured by dominant axis, the region centroids labelled."""
    import plotly.graph_objects as go
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(strands), min(n, len(strands)), replace=False)
    paths = [strands[i] for i in pick]
    fig = go.Figure(data=_lines3d(_by_dominant_axis(paths), ("#d62728", "#2ca02c", "#1f77b4"), name="strands", width=1.5, opacity=0.6) + [_regions(rois)])
    fig.update_layout(scene=_scene(shape), margin=dict(l=0, r=0, t=30, b=0), height=560,
                      title=dict(text=f"{len(paths):,} of {len(strands):,} ground-truth strands, diameters {1e3 * diameters.min():.1f}-{1e3 * diameters.max():.1f} µm (drag to rotate)",
                                 font=dict(size=13)))
    return fig


def matrices(M, score, gt_count):
    """Ours beside the strand-count ground truth on a log colour scale, with the Pearson numbers."""
    plt = _mpl()
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, (title, A) in zip(axes, ((f"streamline counts (Pearson {score['pearson_count']:.3f} vs count, {score['pearson_area']:.3f} vs area)", M),
                                     ("ground truth: strand count", gt_count))):
        im = ax.imshow(np.where(A > 0, A, np.nan), norm=LogNorm(vmin=1, vmax=max(A.max(), 2)), cmap="viridis")
        ax.set_title(title, fontsize=9); ax.set_xticks(range(0, 16, 3)); ax.set_yticks(range(0, 16, 3))
        ax.set_xticklabels(range(1, 17, 3)); ax.set_yticklabels(range(1, 17, 3))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{score['connected_pairs']} of 120 pairs connected, ground truth {score['gt_pairs']}; {score['false_pairs']} false, {score['missed_pairs']} missed", fontsize=9)
    return _image(fig)


def ground_truth_matrix(gt_count, gt_area):
    plt = _mpl()
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, (title, A) in zip(axes, (("strand count", gt_count), ("cross-sectional area", gt_area))):
        im = ax.imshow(np.where(A > 0, A, np.nan), norm=LogNorm(vmin=max(A[A > 0].min(), 1e-3), vmax=A.max()), cmap="viridis")
        ax.set_title(title, fontsize=9); ax.set_xticks(range(0, 16, 3)); ax.set_yticks(range(0, 16, 3))
        ax.set_xticklabels(range(1, 17, 3)); ax.set_yticklabels(range(1, 17, 3))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("the ground truth: 25 connected pairs of 120", fontsize=9)
    return _image(fig)


def timings_rows(seconds, load_seconds):
    rows = [["layout on device (once per process)", f"{load_seconds:.1f}"]]
    return rows + [[k, f"{v:.2f}"] for k, v in seconds.items()]


def floor_slice(floor, z, *, label=""):
    """Axial slice ``z`` of the replay's split-half floor per voxel (the disagreement of the two walker halves,
    the largest over the classes played): what the replay is accurate to, voxel by voxel."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    sl = floor[:, :, z]
    im = ax.imshow(np.where(sl > 0, sl, np.nan).T, origin="lower", cmap="magma", vmin=0, vmax=max(float(np.nanmax(floor)), 1e-6))
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"{label}split-half floor, slice z = {z} (median {np.nanmedian(floor[floor > 0]) if (floor > 0).any() else float('nan'):.4f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return _image(fig)
