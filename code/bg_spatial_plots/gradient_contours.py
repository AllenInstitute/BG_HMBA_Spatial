from dataclasses import dataclass
import sys
sys.path.append("/code")

from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy.spatial import cKDTree

# contour plot functions
def smooth_field(x, y, v, res=120, sigma=4, rng_xy=None,
                 mask_mode="distance", margin_um=None, support=0.05):
    """Bin scattered values to a grid, then NaN-aware Gaussian smooth.
    See curved_gradient_viz.py for the full docstring. sigma is in GRID-CELL
    units; pass rng_xy=None to auto-fit the grid to the data's bounding box.
 
    mask_mode : {"distance", "density", "none"}
        How to blank grid cells outside the tissue.
        "distance" (default): keep grid cells within `margin_um` of a real cell.
            Hugs the true tissue shape with a small, physical margin, so it
            neither clips thinning edges nor balloons a halo past the data. The
            grid is auto-padded so this mask never collides with the box edge
            (which is what produced straight-line clipping).
        "density": blank cells where smoothed count < `support` * its max.
            Simple but clips gradually-thinning borders.
        "none": no masking (field extrapolates past the tissue; debugging only).
    margin_um : float or None
        Distance mask radius in DATA units (microns). If None, defaults to ~2
        grid-bin widths. Smaller hugs the cells tighter; larger fills gaps
        between sparse cells at the cost of extending past the boundary.
    support : float
        Threshold for mask_mode="density" only.
    """
    x = np.asarray(x); y = np.asarray(y); v = np.asarray(v)
    if rng_xy is None:
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    else:
        x0, x1, y0, y1 = rng_xy
    binw = max((x1 - x0) / res, (y1 - y0) / res)
    if margin_um is None:
        margin_um = 2 * binw
    if mask_mode == "distance":                       # pad so the mask edge stays interior
        x0 -= 1.5 * margin_um; x1 += 1.5 * margin_um
        y0 -= 1.5 * margin_um; y1 += 1.5 * margin_um
 
    ex = np.linspace(x0, x1, res + 1); ey = np.linspace(y0, y1, res + 1)
    s, _, _, _ = binned_statistic_2d(x, y, v, 'sum', bins=[ex, ey])
    c, _, _, _ = binned_statistic_2d(x, y, v, 'count', bins=[ex, ey])
    s = s.T; c = c.T
    num = gaussian_filter(np.nan_to_num(s), sigma)
    den = gaussian_filter(c.astype(float), sigma)
    Z = num / np.maximum(den, 1e-9)
    cx = .5 * (ex[:-1] + ex[1:]); cy = .5 * (ey[:-1] + ey[1:])
 
    if mask_mode == "density":
        Z[den < support * den.max()] = np.nan
    elif mask_mode == "distance":
        GXc, GYc = np.meshgrid(cx, cy)
        dist, _ = cKDTree(np.column_stack([x, y])).query(
            np.column_stack([GXc.ravel(), GYc.ravel()]), k=1)
        Z[dist.reshape(GXc.shape) > margin_um] = np.nan
    return np.meshgrid(cx, cy), Z, (cx, cy)

@dataclass
class SpeciesContours:
    levels: np.ndarray
    norm: object
    grids: dict                          # {section_id: (GX, GY, Z)}

    def section_ids(self):
        return list(self.grids)

    def assign_to_obs(self, adata, col='contour_coord', bin_col=None,
                      xcol='x_slab_um', ycol='y_slab_um', sec_col='z_order'):
        """Sample the smoothed field Z at each cell and write it to adata.obs[col].

        For every section in this bundle's grids, the section's smoothed field is
        sampled at the (x, y) of the cells belonging to that section. Cells in
        sections not present in the bundle, or outside the masked grid, get NaN.
        If bin_col is given, also writes the contour-level bin (via self.levels).
        Modifies adata in place; returns adata.
        """
        obs = adata.obs
        vals = np.full(adata.n_obs, np.nan)
        for sec_id, (GX, GY, Z) in self.grids.items():
            mask = (obs[sec_col] == sec_id).to_numpy()
            if not mask.any():
                continue
            cx, cy = GX[0, :], GY[:, 0]
            interp = RegularGridInterpolator((cy, cx), Z, bounds_error=False, fill_value=np.nan)
            pts = np.column_stack([obs[ycol].to_numpy()[mask], obs[xcol].to_numpy()[mask]])
            vals[mask] = interp(pts)
        adata.obs[col] = vals
        if bin_col is not None:
            import pandas as pd
            adata.obs[bin_col] = pd.cut(vals, bins=self.levels, include_lowest=True)
        return adata

def compute_shared_levels(adata, gradient_col, sec_col='z_order',
                          xcol='x_slab_um', ycol='y_slab_um',
                          n_levels=12, level_mode="linear",
                          sigma_um=None, res=120, margin_um=None,
                          cmap="viridis"):
    """Smooth every section of ONE species and compute shared levels.
    Returns a SpeciesContours bundle."""
    grids = {}
    for sec_id, data in adata.obs.groupby(sec_col):
        x = np.asarray(data[xcol]); y = np.asarray(data[ycol])
        sigma = 4 if sigma_um is None else sigma_um / ((x.max() - x.min()) / res)
        (GX, GY), Z, _ = smooth_field(x, y, data[gradient_col].values,
                                      res=res, sigma=sigma, margin_um=margin_um)
        grids[sec_id] = (GX, GY, Z)

    pooled = np.concatenate([Z[np.isfinite(Z)].ravel() for _, _, Z in grids.values()])
    if level_mode == "quantile":
        levels = np.unique(np.nanquantile(pooled, np.linspace(0, 1, n_levels + 1)))
    else:
        levels = np.linspace(pooled.min(), pooled.max(), n_levels + 1)
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N)

    return SpeciesContours(levels=levels, norm=norm, grids=grids)

def draw_section_contour(ax, bundle, sec_id, cmap="viridis",
                         add_colorbar=False, title=None, contour_lw=0.6):
    """Draw ONE section onto an existing ax using the bundle's shared levels/norm."""
    GX, GY, Z = bundle.grids[sec_id]
    cf = ax.contourf(GX, GY, Z, levels=bundle.levels, norm=bundle.norm,
                     cmap=cmap, extend="both")
    ax.contour(GX, GY, Z, levels=bundle.levels, colors="white", linewidths=contour_lw)
    ax.set_aspect("equal")
    #ax.set_title(title if title is not None
                 #else f"section {sec_id}")
    if add_colorbar:
        ax.figure.colorbar(cf, ax=ax, shrink=.8, label="field value (shared bins)")
    return cf

def plot_contours_all_sections(bundle, ncols=3, cmap="viridis"):
    """Plot every section for one species from its bundle."""
    sec_ids = bundle.section_ids()
    nrows = int(np.ceil(len(sec_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.4*nrows), 
                             sharex=True, sharey=True, squeeze=False)
    cf = None
    for ax, sec_id in zip(axes.ravel(), sec_ids):
        cf = draw_section_contour(ax, bundle, sec_id, cmap=cmap)
    for ax in axes.ravel()[len(sec_ids):]:
        ax.axis("off")
    fig.colorbar(cf, ax=axes, shrink=.85, label="field value (shared bins)")
    return fig, axes

def plot_clusters_with_contours(ax, adata, sec_id, bundle, cluster_colors, group_col='Cluster',
                                xcol='x_slab_um', ycol='y_slab_um', sec_col='z_order',
                                fg_size=2, fg_alpha=1.0, bg_size=0.5, bg_alpha=0.5,
                                bg_color='lightgrey',
                                contour_color='k', contour_lw=0.6, contour_alpha=0.8):
    sub = adata[adata.obs[sec_col] == sec_id]
    cats = sub.obs[group_col].astype(str)
    is_fg = cats.isin(cluster_colors).to_numpy()          # highlighted cells

    x = sub.obs[xcol].to_numpy(); y = sub.obs[ycol].to_numpy()

    # 1) background: everything not highlighted -- small + transparent
    ax.scatter(x[~is_fg], y[~is_fg], c=bg_color, s=bg_size, alpha=bg_alpha,
               linewidth=0, rasterized=True)

    # 2) foreground: highlighted clusters -- bigger + opaque, drawn on top
    fg_colors = cats[is_fg].map(cluster_colors).to_numpy()
    ax.scatter(x[is_fg], y[is_fg], c=fg_colors, s=fg_size, alpha=fg_alpha,
               linewidth=0, rasterized=True)

    # 3) contour lines on top of everything
    GX, GY, Z = bundle.grids[sec_id]
    ax.contour(GX, GY, Z, levels=bundle.levels, colors=contour_color,
               linewidths=contour_lw, alpha=contour_alpha)

    ax.set_aspect('equal'); ax.axis('off')
    return ax

def plot_cluster_contour_barplot(ax, adata, sec_id, cluster_colors,
                                 sec_col='z_order', cluster_col='Cluster',
                                 bin_col='contour_bin', mode='abs_counts'):
    sub = adata.obs[(adata.obs[sec_col] == sec_id) &
                    (adata.obs[cluster_col].astype(str).isin(cluster_colors))]
    counts = (sub.groupby([bin_col, cluster_col], observed=False)
                 .size().unstack(cluster_col, fill_value=0))
    counts = counts[[c for c in counts.columns if c in cluster_colors]]
    colors = [cluster_colors[c] for c in counts.columns]

    if mode == 'abs_counts':
        counts.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.9, legend=False)
        ax.set_ylabel('number of cells')
    elif mode == 'composition':                       # each BIN sums to 1
        frac = counts.div(counts.sum(axis=1), axis=0)
        frac.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.9, legend=False)
        ax.set_ylabel('fraction of cells in bin')
    elif mode == 'per_cluster':
        frac = counts.div(counts.sum(axis=0), axis=1)
        for c in frac.columns:
            ax.plot(range(len(frac)), frac[c], color=cluster_colors[c],
                    marker='o', markersize=2, label=c)
        ax.set_ylabel('fraction of cluster in bin')
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of "
                         f"'abs_counts', 'composition', 'per_cluster'")

    ax.set_xlabel('contour bin\n(low → high)')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(range(1, len(counts) + 1), rotation=0)
    return ax