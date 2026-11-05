import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import bg_spatial_plots.figure_utils as futils


def plot_pc_on_single_plane(
    ad_sec,
    spatial_keys,
    pc_index,
    vlim_abs,
    ax,
    marker_size=5,
    cmap='coolwarm',
    invert_y=True,
    show_spines=False,
    rasterized=True
):
    """
    Plot principal component scores on a single spatial plane.

    Parameters
    ----------
    ad_sec : ad.AnnData
        Annotated data object containing spatial coordinates and PC scores.
    spatial_keys : array-like
        Column names or indices for spatial coordinates in ad_sec.obs.
    pc_index : int
        Index of the principal component to plot (0-based).
    vlim_abs : float
        Absolute value limit for colormap scaling (vmin=-vlim_abs, vmax=vlim_abs).
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    marker_size : float, optional
        Size of scatter plot markers. Default is 5.
    cmap : str, optional
        Colormap name. Default is 'coolwarm'.
    invert_y : bool, optional
        If True, invert the y-axis for standard spatial plot orientation.
        Default is True.
    show_spines : bool, optional
        If True, show axis spines. Default is False.

    Returns
    -------
    PathCollection
        Matplotlib scatter plot collection object.
    """
    spatial = ad_sec.obs[list(spatial_keys)].values
    scores = ad_sec.obs[f'PC_{pc_index}']

    sc_plot = ax.scatter(
        spatial[:, 0], 
        spatial[:, 1], 
        c=scores, 
        cmap=cmap, 
        s=marker_size,
        edgecolors='none',
        vmin=-vlim_abs, 
        vmax=vlim_abs,
        rasterized=rasterized
    )
    if invert_y: 
        ax.invert_yaxis()  # Invert y-axis for spatial plots
    futils.format_spatial_axes(ax, show_spines=show_spines, show_axes_labels_ticks=False)

    return sc_plot

def project_2d_points_to_line(points, p1, p2, zero_min_depth=False):
    """
    Project 2D points onto a 1D line.

    Parameters
    ----------
    points : np.ndarray
        The points to project with shape (n_points, 2).
    p1 : np.ndarray
        First point defining the line with shape (2,).
    p2 : np.ndarray
        Second point defining the line with shape (2,).
    zero_min_depth : bool, default=False
        If True, shift projections so the minimum value is 0.

    Returns
    -------
    np.ndarray
        Distances of points along the projected line with shape (n_points,).
    """   
    line_vec = p2 - p1
    line_unitvec = line_vec / np.linalg.norm(line_vec)
    vec_from_p1 = points - p1
    proj_lengths = np.dot(vec_from_p1, line_unitvec)

    if zero_min_depth:
        # shift so the smallest projection is 0
        # otherwise it may be negative for points "before" p1
        proj_lengths -= proj_lengths.min()
    
    return proj_lengths

def find_best_gradient_pc(xpca, projected_depth):
    """
    Identify the PC with the highest correlation to projected depth.

    Parameters
    ----------
    xpca : np.ndarray
        PCA scores array with shape (n_cells, n_pcs).
    projected_depth : np.ndarray
        1D array of projected depth values with shape (n_cells,).

    Returns
    -------
    int
        Index of the PC with highest absolute correlation to projected depth.
    """
    pc_corrs = [np.corrcoef(xpca[:, i], projected_depth)[0, 1] for i in range(xpca.shape[1])]
    best_pc = np.argmax(np.abs(pc_corrs))
    print(f"PC{best_pc+1} (index: {best_pc}) has the highest correlation to projected depth: {pc_corrs[best_pc]:.3f}")
    return best_pc