"""The page's views: a DWI slice with the FODs' principal directions drawn over it, the replay floor slice, scalar and
fraction maps, the tractogram and the ground-truth strands as rotatable 3-D figures with the region markers, the
connectome beside its truth for any number of regions, the spread over repeated runs, the response curves, the
timings. Matplotlib for the slices and matrices (PIL images), Plotly for the 3-D views (figures the page rotates).
Every number drawn comes in from :mod:`space.pipeline`, a source or the page."""
from __future__ import annotations

import io

import numpy as np

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


NAMED_TICKS = 20                                     # up to this many regions the ticks carry the names


def pair_panel(panels, suptitle, *, vmin=1.0, names=None):
    """Square matrices side by side on a log colour scale (zeros blank): ``panels`` is ``[(title, M), ...]``. The
    ticks are the regions' names when ``names`` has at most :data:`NAMED_TICKS`, else every region number at a
    stride that keeps about ten ticks."""
    plt = _mpl()
    from matplotlib.colors import LogNorm
    n = len(panels[0][1])
    big = n > 16
    fig, axes = plt.subplots(1, len(panels), figsize=((6.2 if big else 4) * len(panels), 5.8 if big else 3.8))
    labelled = names is not None and len(names) <= NAMED_TICKS
    ticks = range(n) if labelled else range(0, n, max(1, int(np.ceil(n / 10))) if big else 3)
    for ax, (title, A) in zip(np.atleast_1d(axes), panels):
        A = np.asarray(A, np.float64)
        top = max(float(A.max()), vmin * 2)
        im = ax.imshow(np.where(A > 0, A, np.nan), norm=LogNorm(vmin=vmin, vmax=top), cmap="viridis")
        ax.set_title(title, fontsize=9); ax.set_xticks(list(ticks)); ax.set_yticks(list(ticks))
        if labelled:
            ax.set_xticklabels(names, rotation=90, fontsize=6); ax.set_yticklabels(names, fontsize=6)
        else:
            ax.set_xticklabels([t + 1 for t in ticks], fontsize=7); ax.set_yticklabels([t + 1 for t in ticks], fontsize=7)
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


def region_markers(regions):
    """The region centroids as Plotly markers (built once per source; the 3-D views take it): ``regions`` a
    :class:`space.pipeline.Regions` (centroids through its affine, into the tracking frame) or a label image (voxel
    coordinates); numbered while there are at most :data:`NAMED_TICKS` regions, else coloured by their group with the
    name on hover."""
    import plotly.graph_objects as go
    labels = getattr(regions, "labels", regions)
    affine = np.asarray(getattr(regions, "affine", np.eye(4)), np.float64)
    names = getattr(regions, "names", None)
    n = len(names) if names is not None else int(np.max(labels))
    group_of = {}
    for g, (_, ids) in enumerate(getattr(regions, "groups", ()) or ()):
        for i in ids:
            group_of[int(i)] = g
    cs = []
    for k in range(1, n + 1):
        ijk = np.argwhere(labels == k)
        if len(ijk):
            cs.append((k, ijk.mean(0) @ affine[:3, :3].T + affine[:3, 3]))
    few = n <= NAMED_TICKS
    colour = [REGION_COLOURS[(group_of.get(c[0], c[0] - 1)) % len(REGION_COLOURS)] for c in cs]
    return go.Scatter3d(x=[c[1][0] for c in cs], y=[c[1][1] for c in cs], z=[c[1][2] for c in cs], mode="markers+text" if few else "markers",
                        text=[str(c[0]) if few or names is None else names[c[0] - 1] for c in cs], textposition="top center",
                        hoverinfo="text", marker=dict(size=5 if few else 3, color=colour), name="regions", showlegend=False)


def _figure3d(paths, regions, shape, title, *, name, width, opacity):
    """Paths coloured by dominant axis with the region markers: in a cube of ``shape`` voxels, or in the box
    ``shape = ((x0, x1), (y0, y1), (z0, z1))`` in millimetres."""
    import plotly.graph_objects as go
    box = np.asarray(shape, np.float64)
    ranges, unit, aspect = (box, "mm", "data") if box.ndim == 2 else (np.stack([np.zeros(3), box], 1), "voxels", "cube")
    fig = go.Figure(data=_lines3d(_by_dominant_axis(paths), AXIS_COLOURS, name=name, width=width, opacity=opacity) + [regions])
    fig.update_layout(scene=dict(xaxis=dict(range=list(ranges[0]), title=f"x ({unit})"), yaxis=dict(range=list(ranges[1]), title="y"),
                                 zaxis=dict(range=list(ranges[2]), title="z"), aspectmode=aspect),
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
    return pair_panel([(f"streamline counts (Pearson {score['pearson_count']:.3f} vs count, {score['pearson_area']:.3f} vs area)", M),
                        ("ground truth: strand count", gt_count)],
                       f"{score['connected_pairs']} of {len(M) * (len(M) - 1) // 2} pairs connected, ground truth {score['gt_pairs']}; "
                       f"{score['false_pairs']} false, {score['missed_pairs']} missed")


def ground_truth_matrix(gt_count, gt_area, n_pairs):
    """The two ground-truth matrices; ``n_pairs`` is how many of the 120 pairs the strands connect."""
    return pair_panel([("strand count", gt_count), ("cross-sectional area", gt_area)],
                       f"the ground truth: {n_pairs} connected pairs of {len(gt_count) * (len(gt_count) - 1) // 2}", vmin=max(float(gt_area[gt_area > 0].min()), 1e-3))


def spread_matrices(spread):
    """The mean streamline count per pair beside its standard deviation over the repeated runs."""
    return pair_panel([(f"mean count over {spread['n']} keys", spread["mean"]), ("standard deviation over the keys", spread["std"])],
                       f"{spread['key']} {spread['pearson_mean']:.3f} ± {spread['pearson_std']:.3f}; median pair CV {spread['cv_median']:.2f}; "
                       f"{spread['pairs_always']} pairs in every run, {spread['pairs_any']} in any")


def timings_rows(seconds, load_seconds):
    """The stage times as table rows, the source's load time (once per process) first."""
    rows = [["source loaded and warmed (once per process)", f"{load_seconds:.1f}"]]
    return rows + [[k, f"{v:.2f}"] for k, v in seconds.items()]


def map_slice(vol, z, title, *, cmap="viridis", symmetric=False, vmin=None, vmax=None, unit=""):
    """Axial slice ``z`` of a scalar map (NaN blank): a diverging scale around zero when ``symmetric`` (the
    difference maps), else the map's own range unless ``vmin`` / ``vmax`` fix it."""
    fig, ax = _slice_axes(title)
    sl = np.asarray(vol[:, :, z], np.float64)
    finite = sl[np.isfinite(sl)]
    if symmetric:
        top = float(np.max(np.abs(finite))) if finite.size else 1.0
        top = top or 1.0
        im = ax.imshow(sl.T, origin="lower", cmap="RdBu_r", vmin=-top, vmax=top)
    else:
        lo = float(np.min(finite)) if (vmin is None and finite.size) else (vmin or 0.0)
        hi = float(np.max(finite)) if (vmax is None and finite.size) else (vmax if vmax is not None else 1.0)
        im = ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
    fig.colorbar(im, ax=ax, fraction=0.046, label=unit)
    return _image(fig)


def grid_box(shape, affine):
    """The millimetre box ``((x0, x1), (y0, y1), (z0, z1))`` the grid of ``shape`` covers under ``affine``: the 3-D
    views' extent for a source whose tracking frame is not its voxel frame."""
    corners = np.array(np.meshgrid(*[[-0.5, n - 0.5] for n in shape[:3]], indexing="ij")).reshape(3, -1).T
    w = corners @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]
    return tuple((float(a), float(b)) for a, b in zip(w.min(0), w.max(0)))


def regions3d(regions, shape):
    """The regions' centroids alone in 3-D, coloured by group, in the tracking frame's box."""
    return _figure3d([], region_markers(regions), grid_box(shape, regions.affine), f"{regions.n} regions (hover for the name; colour: lobar group)",
                     name="regions", width=1.0, opacity=1.0)


def _rgb(fractions, z):
    """An axial slice of three fractions as red / green / blue."""
    return np.clip(np.nan_to_num(np.asarray(fractions, np.float64)[:, :, z, :3]), 0, 1).transpose(1, 0, 2)


def fractions_pair(recovered, truth, z):
    """The recovered fractions beside the input's at slice ``z`` (WM red, GM green, CSF blue), and their difference
    per tissue; the recovered side says so when the reconstruction estimated none."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 5 if recovered is not None else 1, figsize=(15 if recovered is not None else 4, 3.4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(_rgb(truth, z), origin="lower"); axes[0].set_title(f"input fractions, z = {z} (WM r, GM g, CSF b)", fontsize=8)
    if recovered is not None:
        axes[1].imshow(_rgb(recovered, z), origin="lower"); axes[1].set_title("recovered (multi-tissue CSD)", fontsize=8)
        for k, t in enumerate(("WM", "GM", "CSF")):
            d = np.asarray(recovered, np.float64)[:, :, z, k] - np.asarray(truth, np.float64)[:, :, z, k]
            im = axes[2 + k].imshow(d.T, origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
            axes[2 + k].set_title(f"{t}: recovered − input", fontsize=8)
        fig.colorbar(im, ax=axes[-1], fraction=0.046)
    else:
        axes[0].set_title(f"input fractions, z = {z}: the single-tissue reconstruction estimates none", fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    return _image(fig)


def input_slices(b0, peaks, amp, fractions, labels, z):
    """The input at slice ``z``: the mean b = 0 with the WM FOD's principal peaks, the fractions (WM r, GM g, CSF b),
    the parcellation."""
    plt = _mpl()
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    sl = np.nan_to_num(np.asarray(b0, np.float64)[:, :, z])
    axes[0].imshow(sl.T, origin="lower", cmap="gray", vmin=0, vmax=float(np.quantile(sl[sl > 0], 0.99)) if (sl > 0).any() else 1)
    d = peaks[:, :, z]; a = amp[:, :, z]
    ii, jj = np.nonzero(a > 0)
    if len(ii):
        L = 0.45 * a[ii, jj] / a[ii, jj].max()
        seg = np.stack([np.stack([ii - L * d[ii, jj, 0], jj - L * d[ii, jj, 1]], 1), np.stack([ii + L * d[ii, jj, 0], jj + L * d[ii, jj, 1]], 1)], 1)
        axes[0].add_collection(LineCollection(seg, colors=np.abs(d[ii, jj]), linewidths=0.8))
    axes[0].set_title(f"mean b = 0 and the WM FOD's principal peaks, z = {z}", fontsize=8)
    axes[1].imshow(_rgb(fractions, z), origin="lower"); axes[1].set_title("fractions (WM r, GM g, CSF b)", fontsize=8)
    lab = np.asarray(labels)[:, :, z].astype(float)
    axes[2].imshow(np.where(lab > 0, lab, np.nan).T, origin="lower", cmap="tab20", interpolation="nearest")
    axes[2].set_title(f"regions ({int(np.max(labels))})", fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    return _image(fig)


def response_curves(true, estimated, note=""):
    """The packs' exact response at the run's tissue (solid) against the reconstruction's estimate from the replayed
    data (dashed): the WM response of a fibre against the gradient's angle per shell, and GM's and CSF's signal per
    shell, each over its b = 0 value; a tissue the reconstruction did not estimate is drawn true only."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    if true is None:
        for ax in axes:
            ax.set_axis_off()
        axes[0].set_title("no diffusion-weighted shell: no response to draw", fontsize=9)
        return _image(fig)
    cmap = plt.get_cmap("viridis")
    b = np.asarray(true["b"], np.float64)
    for k, bb in enumerate(b):
        col = cmap(k / max(len(b) - 1, 1))
        axes[0].plot(true["angles"], true["wm"][k], "-", color=col, label=f"b = {bb:g}")
        if estimated is not None and estimated.get("wm") is not None:
            axes[0].plot(estimated["angles"], estimated["wm"][k], "--", color=col)
    axes[0].set_xlabel("angle between the gradient and the fibre (°)"); axes[0].set_ylabel("S / S(b = 0)")
    axes[0].set_title("WM: the pack's exact response (solid) vs the estimate (dashed)", fontsize=9); axes[0].legend(fontsize=7)
    for t, marker in (("gm", "o"), ("csf", "s")):
        axes[1].plot(b, true[t], "-" + marker, label=f"{t.upper()} exact")
        if estimated is not None and estimated.get(t) is not None:
            axes[1].plot(b, estimated[t], "--" + marker, mfc="none", label=f"{t.upper()} estimated")
    axes[1].set_yscale("log"); axes[1].set_xlabel("b (s/mm²)"); axes[1].set_ylabel("S / S(b = 0)")
    axes[1].set_title("GM and CSF" + (f" ({note})" if note else ""), fontsize=9); axes[1].legend(fontsize=7)
    fig.tight_layout()
    return _image(fig)
