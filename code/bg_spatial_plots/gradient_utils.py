import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import bg_spatial_plots.figure_utils as futils
import pathlib

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


def run_pca_on_spatial_cells(
    adata,
    layer='log2cpt',
    pca_n_comps=50,
    normalize=False,
    scale=True
):
    """
    Run PCA on spatial transcriptomic cells.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial transcriptomic data.
    layer : str, default='log2cpt'
        Layer to calculate PCA on. Use None if normalizing within this function 
        (although this will update counts in X which is not always desired).
    pca_n_comps : int, default=50
        Number of PCA components to calculate.
    normalize : bool, default=False
        Whether to normalize the data before running PCA.
    scale : bool, default=True
        Whether to scale data to unit variance and zero mean.

    Returns
    -------
    None
        Updates the adata object in-place with PCA results.
    """

    # Normalize and log-transform
    if normalize:
        assert layer is None, "If normalizing within this function, layer must be None"
        sc.pp.normalize_total(adata, target_sum=1e3)
        sc.pp.log1p(adata)
    
    if scale:
        from sklearn.preprocessing import StandardScaler
        # add new layer so as not to scale in place (could affect other calculations)
        if not layer:
            scaled_data = StandardScaler().fit_transform(adata.X)
            layer = 'X_scaled'
        else:
            if type(adata.layers[f'{layer}']) is not np.ndarray:
                scaled_data = StandardScaler().fit_transform(adata.layers[f'{layer}'].toarray())
            else:
                scaled_data = StandardScaler().fit_transform(adata.layers[f'{layer}'])

            layer = f'{layer}_scaled'
        
        adata.layers[layer] = scaled_data

    # Run PCA
    sc.tl.pca(adata, svd_solver='arpack', n_comps=pca_n_comps, layer=layer, dtype='float64', use_highly_variable=False)



def estimate_global_gradient(points, values):
    """
    Estimate the best-fit (global) gradient of a scalar field over the entire 2D point cloud.
    Uses linear regression over all points.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of points with shape (N, 2).
    values : np.ndarray
        Scalar values at each point with shape (N,).

    Returns
    -------
    np.ndarray
        Estimated global gradient of shape (2,).
    """
    X = points - points.mean(axis=0)
    y = values - values.mean()
    grad, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    return grad

def estimate_local_gradients(points, values, grid_n=None, grid_spacing=100.,
                             min_points=5, radius_factor=1.0,
                             weight_by_n=False, grid_point_distance=None):
    """
    Estimate the local gradient of a scalar field at each point on a 2D grid.
    
    Calculates the gradient of a scalar field (e.g., gene expression values) at each 
    point on a 2D grid by performing a local least-squares fit of the gradient using 
    nearby points within a specified radius.

    Parameters
    ----------
    points : np.ndarray
        Coordinates of the points with shape (N, 2).
    values : np.ndarray
        Scalar values at each point with shape (N,).
    grid_n : int, optional
        Number of grid points along each axis. If provided, the grid will be evenly 
        spaced between the minimum and maximum coordinates of the points.
    grid_spacing : float, default=100.
        The spacing between grid points. Ignored if `grid_n` is provided.
    min_points : int, default=5
        Minimum number of points required within the radius to estimate the gradient.
        If fewer points are available, the gradient at that grid point is set to [0, 0].
    radius_factor : float, default=1.0
        Factor to scale the radius for local gradient estimation.
    weight_by_n : bool, default=False
        Whether to weight gradients by the number of points used in estimation.
    grid_point_distance : float, optional
        Maximum distance from grid point to nearest data point. If the nearest point 
        is farther than this distance, the gradient is set to [0, 0].

    Returns
    -------
    tuple
        - gradients : np.ndarray
            Local gradients at each grid point with shape (grid_y_size, grid_x_size, 2).
        - grid_x : np.ndarray
            X-coordinates of grid points.
        - grid_y : np.ndarray
            Y-coordinates of grid points.
    """
    
    if radius_factor < 1/np.sqrt(2):
        print("Warning: radius_factor < 1/sqrt(2) may lead to some areas being missed.")
    
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    if grid_n is not None:
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, grid_n),
            np.linspace(y_min, y_max, grid_n)
        )
        grid_spacing = np.diff(grid_x)[0][0]
    else:
            grid_x, grid_y = np.meshgrid(
            np.arange(x_min, x_max, grid_spacing),
            np.arange(y_min, y_max, grid_spacing)
        )

    gradients = np.zeros((grid_x.shape[0], grid_y.shape[1], 2))

    for i in range(grid_x.shape[0]):
        for j in range(grid_y.shape[1]):
            grid_point = np.array([grid_x[i, j], grid_y[i, j]])
            distances = np.linalg.norm(points - grid_point, axis=1)

            # limit data to cells within radius of grid_spacing*radius_factor
            grid_cells = points[distances<=radius_factor*grid_spacing]
            grid_values= values[distances<=radius_factor*grid_spacing]
            
            # Check if there are enough points contributing to the gradient
            contributing_points = grid_cells.shape[0]
            if contributing_points < min_points:
                gradients[i, j] = [0, 0]
                continue

            X = grid_cells - np.mean(grid_cells, axis=0)
            y = grid_values -np.mean(grid_values, axis =0) # - (weights @ values)
            grad, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if weight_by_n:
                grad = grad * contributing_points
            gradients[i, j] = grad

    # run through grid points and exclude any that are too far from any data point
    if grid_point_distance is not None:
        for i in range(grid_x.shape[0]):
            for j in range(grid_y.shape[1]):
                grid_point = np.array([grid_x[i, j], grid_y[i, j]])
                distances = np.linalg.norm(points - grid_point, axis=1)
                if np.min(distances) > grid_point_distance:
                    gradients[i, j] = [0, 0]

    return gradients, grid_x, grid_y





def add_pcs_to_obs(adata, pc_indices=None):
    """
    Add PCA scores from obsm as obs columns.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object. Must have PCA run with results in obsm.
    pc_indices : list, optional
        Which PC indices to add to obs. Starts counting from 0.
        If None, all PCs are added.

    Returns
    -------
    None
        Updates adata.obs in-place with PC score columns.
    """
    if not pc_indices:
        # add all pcs
        pc_indices = np.arange(0, adata.obsm['X_pca'].shape[1])
    else:
        pc_indices = [pc for pc in pc_indices]
    pc_scores = adata.obsm['X_pca'][:, pc_indices]
    colnames = [f'PC_{pc_ind}' for pc_ind in pc_indices]
    # alt format: colnames = [f"pca{str(i).zfill(3)}" for pc_ind in pc_indices]
    adata.obs[colnames] = pc_scores

def add_pc_gradient_arrow_to_section(
    ax, 
    coords_sec, 
    pc_scores_sec, 
    arrow_scale=0.3,
    arrow_color='black',
    arrow_linewidth=2,
    arrow_alpha=1
):
    """
    Add a gradient direction arrow to an axis showing the direction of maximum PC variation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw the arrow on.
    coords_sec : np.ndarray
        Nx2 array of spatial coordinates for the section.
    pc_scores_sec : np.ndarray
        N-length array of PC scores for a single PC in the section.
    arrow_scale : float, default=0.3
        Scale factor for arrow length (relative to data range).
    arrow_color : str, default='black'
        Color of the gradient arrow.
    arrow_linewidth : float, default=2
        Width of the gradient arrow.
    arrow_alpha : float, default=1
        Transparency of the arrow.

    Returns
    -------
    None
    """
    # Calculate gradient direction for this section
    gradient_direction = estimate_global_gradient(coords_sec, pc_scores_sec)
    
    # Calculate arrow length and position
    coord_range = np.ptp(coords_sec, axis=0)
    arrow_length = arrow_scale * np.mean(coord_range)
    center = coords_sec.mean(axis=0)
    
    # Draw gradient arrow
    start = (
        center[0] - arrow_length * gradient_direction[0] / 2,
        center[1] - arrow_length * gradient_direction[1] / 2
    )

    end = (
        center[0] + arrow_length * gradient_direction[0] / 2,
        center[1] + arrow_length * gradient_direction[1] / 2
    )

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            color=arrow_color,
            linewidth=arrow_linewidth,
            alpha=arrow_alpha,
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
        )
    )

def add_pc_local_gradient_arrows_to_section(
    ax,
    section_coords,
    section_pc_scores,
    grid_spacing=250,
    min_points=30,
    radius_factor=1.5,
    local_gradient_scale=1/300,
    arrow_color='black',
    arrow_alpha=0.7,
    weight_by_n=True,
    grid_point_distance=None
):
    """
    Add local gradient arrows to an axis showing the local direction of PC variation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw the arrows on.
    section_coords : np.ndarray
        Nx2 array of spatial coordinates for the section.
    section_pc_scores : np.ndarray
        N-length array of PC scores for a single PC in the section.
    grid_spacing : float, default=250
        Spacing between grid points where gradients are estimated.
    min_points : int, default=30
        Minimum number of points required to estimate gradient at a grid point.
    radius_factor : float, default=1.5
        Factor to multiply grid spacing to define neighborhood radius.
    local_gradient_scale : float, default=1/300
        Scale factor for arrow lengths.
    arrow_color : str, default='black'
        Color of the gradient arrows.
    arrow_alpha : float, default=0.7
        Transparency of the arrows.
    weight_by_n : bool, default=True
        Whether to weight gradient estimates by number of points used.
    grid_point_distance : float or None, default=None
        Minimum distance between grid points. If None, defaults to grid_spacing / 2.

    Returns
    -------
    None
    """
    
    # Calculate local gradients using the existing function
    gradients, grid_x, grid_y = estimate_local_gradients(
        section_coords,
        section_pc_scores,
        grid_spacing=grid_spacing,
        min_points=min_points,
        radius_factor=radius_factor,
        weight_by_n=weight_by_n,
        grid_point_distance=grid_point_distance
    )
    
    # Add quiver plot to the axis
    ax.quiver(
        grid_x, grid_y,
        gradients[:, :, 0], gradients[:, :, 1],
        angles="xy", 
        scale_units="xy", 
        scale=local_gradient_scale, 
        color=arrow_color, 
        alpha=arrow_alpha
    )

def extract_projection_landmarks(heatmap_array, genes):
    """
    Compute half-max crossing, peak location, and direction for each gene.

    Parameters
    ----------
    heatmap_array : pd.DataFrame
        Array of gene expression values with genes as rows and distance bins as columns.
    genes : list of str
        List of gene names corresponding to rows in heatmap_array.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'gene', 'crossing', 'peak', and 'direction' for each gene.
    """
    records = []

    for i, gene in enumerate(genes):
        values = heatmap_array.loc[gene, :]

        # note: crossing and direction can flip depending on the binsize
        crossing = np.argmin(np.abs(values - 0.5))
        peak = np.argmax(values)
        direction = peak > crossing

        records.append({
            "gene": gene,
            "crossing": crossing,
            "peak": peak,
            "direction": direction
        })

    return pd.DataFrame.from_records(records)



def create_heatmap_array(heatmap_data, cols_to_plot, n_bins, proj_col_name='proj_lengths', normalize=True):
    """
    Bin spatial data values by distance along a projected line.

    Parameters
    ----------
    heatmap_data : pd.DataFrame
        Data projected to line, including the projection column.
    cols_to_plot : list
        Column names to bin (e.g., PCs or gene expression values).
    n_bins : int
        Number of distance bins to create.
    proj_col_name : str, default='proj_lengths'
        Column containing distance along the projected line.
    normalize : bool, default=True
        Whether to minmax normalize the data in cols_to_plot per column.

    Returns
    -------
    tuple
        - heatmap_array : pd.DataFrame
            Binned and averaged data with genes as rows and bins as columns.
        - bin_centers : np.ndarray
            Center positions of each bin.
        - bin_cell_counts: np.ndarray
            cells per bin
    """
    # Bin the distance values
    #adata_thresholded = adata.obs[adata.obs['proj_lengths'] > 0]
    #heatmap_data['distance_bin'] = pd.cut(adata_thresholded['proj_lengths'], bins=n_bins)
    heatmap_data['distance_bin'] = pd.cut(heatmap_data[proj_col_name], bins=n_bins)
    bin_cell_counts = heatmap_data.groupby('distance_bin').count().values[:,0]
    heatmap_data_binned = heatmap_data.groupby('distance_bin').mean()

    # minmax normalize binned counts
    heatmap_subset = heatmap_data_binned.loc[:, cols_to_plot]

    if normalize:
        min_params = heatmap_subset[cols_to_plot].min()
        max_params = heatmap_subset[cols_to_plot].max()
        heatmap_subset[cols_to_plot] = (heatmap_subset[cols_to_plot] - min_params) / (max_params - min_params)

    heatmap_array = heatmap_subset.T
    bin_centers = heatmap_array.columns.values.categories.mid
    heatmap_array.columns = bin_centers

    return heatmap_array, bin_centers, bin_cell_counts


def plot_cells_and_line(section_ad, line, x_col_name, y_col_name, ax=None, **kwargs):
    """
    Plot cells as scatter points and a line on a matplotlib Axes.

    Parameters
    ----------
    section_ad : anndata.AnnData
        Object containing coordinates in .obs.
    line : array-like, shape (2, 2)
        Two points defining the line to draw.
    x_col_name : str
        Column name for x-coordinates in section_ad.obs.
    y_col_name : str
        Column name for y-coordinates in section_ad.obs.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    **kwargs : dict
        Extra arguments passed to ax.scatter (e.g., s, c, alpha, etc.).

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    x = section_ad.obs[x_col_name].values
    y = section_ad.obs[y_col_name].values

    if not ax:
        _, ax = plt.subplots(figsize=(5,5))

    ax.scatter(
        x,
        y,
        **kwargs   # pass color, size (s), alpha, etc.
    )
    ax.plot(line[:, 0], line[:, 1], "r-")
    ax.set_aspect("equal", adjustable="box")  # maybe unnecessary

    return ax

def plot_gene_depth_heatmap(
    adata, 
    genes, 
    n_bins, 
    min_cells = 0,
    max_bins_to_keep =None,
    layer=None,
    proj_col_name='proj_lengths', 
    normalize=True, 
    ordering_method=None, 
    show_summary_points=None, 
    palette='magma', 
    scatter_kwargs={'color': 'white', 's': 30, 'marker': 'o'},
    ax=None, pad_gene_names = None,
):
    """
    Plot binned gene expression as a heatmap versus projected distance.

    Parameters
    ----------
    adata : anndata.AnnData
        The spatial dataset.
    genes : list of str
        Gene names to include in the heatmap.
    n_bins : int
        Number of distance bins for binning expression values.
    proj_col_name : str, default='proj_lengths'
        Column in adata.obs containing projected distance values.
    normalize : bool, default=True
        Whether to minmax normalize expression values per gene.
    ordering_method : str or None, default=None
        Method to order genes. Options: 'half-max', 'peak', or None (no reordering).
    show_summary_points : str or None, default=None
        Which landmarks to show as scatter points. Options: 'crossing', 'peak', or None.
    palette : str, default='magma'
        Colormap name for the heatmap.
    scatter_kwargs : dict, default={'color': 'white', 's': 30, 'marker': 'o'}
        Keyword arguments for scatter plot of summary points.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new axes.

    Returns
    -------
    tuple
        - ax : matplotlib.axes.Axes
            The axes containing the heatmap.
        - summary : pd.DataFrame
            DataFrame with landmark information for each gene.
    """

    

    if not ax:
        _, ax = plt.subplots(figsize=(5,4))
    print(type(genes[0]))
    heatmap_data = adata[:,genes].to_df(layer=layer).join(adata.obs[proj_col_name])
    heatmap_array, bin_centers, bin_counts = create_heatmap_array(heatmap_data, genes, proj_col_name=proj_col_name, n_bins=n_bins, normalize=normalize)
    print(heatmap_array.shape)
    print(bin_counts.shape)
    print(bin_centers.shape)
    print(type(heatmap_array))
    heatmap_array = heatmap_array.loc[:,bin_counts.flat>=min_cells]

    print("after subsetting")
    print(heatmap_array.shape)
    bin_centers = bin_centers[bin_counts>=min_cells]

    # cut off both ends to reach the max_bins_to_keep
    if max_bins_to_keep is not None:
        bin_diff = heatmap_array.shape[1] - max_bins_to_keep
        if bin_diff >0:
            one_end = int(np.floor(bin_diff/2))
            the_other_end = bin_diff-one_end
            heatmap_array = heatmap_array.iloc[:,one_end:-the_other_end].copy()
            bin_centers=bin_centers[one_end:-the_other_end]
    
    summary = extract_projection_landmarks(heatmap_array, genes)

    if ordering_method:
        if ordering_method =="half-max":
            crossing_df = summary.sort_values(["direction","crossing"])
        elif ordering_method == "peak":
            crossing_df = summary.sort_values(["direction","peak"])
        else:
            raise ValueError("ordering keyword argument takes values `half-max` or `peak`")
        
        # reorder array and genes list
        ordering = crossing_df.index.values
        genes = [genes[i] for i in ordering]
        heatmap_array = heatmap_array.reindex(genes)
        summary = summary.reindex(ordering)

    ax.imshow(heatmap_array, cmap=palette)
    if pad_gene_names is not None:
        gene_names_to_plot = [g.rjust(pad_gene_names) for g in genes]
    else:
        gene_names_to_plot = genes

    ax.set_yticks(range(len(genes)), gene_names_to_plot)
    ax.set_xlabel("projected distance (normalized)")

    if show_summary_points:
        assert(show_summary_points in ['crossing', 'peak'])
        rows = np.arange(0, len(summary))
        cols = summary[show_summary_points].values

        # Plot points at cell centers
        ax.scatter(
            cols,
            rows,
            **scatter_kwargs
        )

    return ax, summary, bin_centers





def get_top_pc_genes(adata, pc_index, top_n=10, kind='both'):
    """
    Get the top genes contributing the most to a selected PC.

    Parameters
    ----------
    adata : anndata.AnnData
        The spatial dataset. Assumes PC loadings stored in adata.varm['PCs'].
    pc_index : int
        Which PC to analyze. Starts from 0.
    top_n : int, default=10
        The number of top genes to return.
    kind : str, default='both'
        Whether to return genes with highest absolute value loadings ('both'),
        or just highest positive ('positive') or negative ('negative') loadings.

    Returns
    -------
    Index
        Gene names of the top genes for the specified PC.
    """
    loadings = adata.varm['PCs'][:, pc_index]
    if kind == 'both':
        # get genes with highest absolute value loadings
        top_indices = np.absolute(loadings).argsort()[::-1][:top_n]
    elif kind == 'positive':
        # get top positive genes
        top_indices = loadings.argsort()[::-1][:top_n]
    elif kind == 'negative':
        # get top negative genes
        top_indices = loadings.argsort()[:top_n]
    else:
        raise ValueError("kind must be 'both', 'positive', or 'negative'")

    return adata.var_names[top_indices]




def bin_points_by_percentile(xy, scalar_values, n_bins=5):
    """
    Bin points based on percentiles of a scalar dimension and compute centroids.

    Parameters
    ----------
    xy : (N, 2) array-like
        2D coordinates of points
    scalar_values : (N,) array-like
        Scalar values for each point
    n_bins : int
        Number of bins (default 5 for quintiles)

    Returns
    -------
    centroids : (n_bins, 2) array
        Centroid of each bin
    bin_labels : (n_bins,) array of strings
        Labels for each bin (e.g., "0-20%", "20-40%", etc.)
    bin_indices : list of arrays
        Indices of points in each bin
    """
    xy = np.asarray(xy, dtype=float)
    scalar_values = np.asarray(scalar_values, dtype=float)

    if xy.shape[0] != len(scalar_values):
        raise ValueError("xy and scalar_values must have same length")

    # Percentile edges
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(scalar_values, percentiles)

    centroids = []
    bin_labels = []
    bin_indices = []

    for i in range(n_bins):
        if i == 0:
            mask = scalar_values <= edges[i + 1]
        elif i == n_bins - 1:
            mask = scalar_values >= edges[i]
        else:
            mask = (scalar_values >= edges[i]) & (scalar_values <= edges[i + 1])

        if np.any(mask):
            centroid = xy[mask].mean(axis=0)
            centroids.append(centroid)
            bin_indices.append(np.where(mask)[0])
            label = f"{percentiles[i]:.0f}-{percentiles[i+1]:.0f}%"
            bin_labels.append(label)

    return np.array(centroids), np.array(bin_labels), bin_indices


def get_pc_spatial_spread(df_with_pcs, multi_section_column = None, multi_section_values = [], pc_columns=[],
                        spatial_obs_columns = ["x","y"], percentile_bins = 5, plot_stuff= False, savefig_path= None, slices_to_save=[]):   
    
    # defaults to analyzing first 10 columns with column names starting with"PC"
    DEFAULT_PCs_TO_RUN = 10
    all_centroid_records=[]
    
    # only save figures if plotting stuff
    if not plot_stuff:
        savefig_path=None
    
    if not multi_section_column:
        multi_section_column = "no column chosen"
        multi_section_values = [None]
    elif len(multi_section_values)==0:
        raise ValueError("please supply a list of values for column "+multi_section_column)



    # loop through sections
    for value  in multi_section_values:
        if not value:
            df_for_centroids = df_with_pcs
        else:
            df_for_centroids = df_with_pcs.loc[df_with_pcs[multi_section_column]==value,:]



        if len(pc_columns) ==0:
            pc_columns = [c for c in df_for_centroids.columns if c.startswith("PC")]
            if len(pc_columns)>DEFAULT_PCs_TO_RUN:
                pc_columns = pc_columns[:DEFAULT_PCs_TO_RUN]





        if plot_stuff:
            fig, axes = plt.subplots(3, 3, figsize=(14, 10))
            axes = axes.flatten()

        #within each section, loop through the PCs, adding to the records for the output dataframe
        for idx, dim in enumerate(pc_columns):

            # Bin points by percentile and compute centroids
            xy_data = df_for_centroids[[spatial_obs_columns[0], spatial_obs_columns[1]]].to_numpy()
            centroids, bin_labels, _ = bin_points_by_percentile(
                xy_data, df_for_centroids[dim].to_numpy(), n_bins=5
            )



            centroid_record = {bin_labels[ii]+"_x": centroids[ii][0] for ii in range(len(bin_labels))}
            centroid_record.update( {bin_labels[ii]+"_y": centroids[ii][1] for ii in range(len(bin_labels))})
            centroid_record.update(dict(first_last_vector= centroids[-1]-centroids[0]))
            centroid_record.update(dict(first_last_distance= np.linalg.norm(centroids[-1]-centroids[0])))

            centroid_record.update(dict(multi_section_column=value,pc=dim))
            all_centroid_records.append(centroid_record)
            
            
            if plot_stuff:

                if idx >= len(axes):
                    print(idx)
                    print(len(axes))
                    continue
                ax = axes[idx]

                # Scatter plot colored by scalar values
                scatter = ax.scatter(
                    df_for_centroids[spatial_obs_columns[0]],
                    df_for_centroids[spatial_obs_columns[1]],
                    c=df_for_centroids[dim],
                    cmap="viridis",
                    s=2,
                    alpha=0.6,
                )
                
                
                # Plot centroids with labels
                ax.scatter(
                    centroids[:, 0],
                    centroids[:, 1],
                    s=200,
                    c="red",
                    marker="*",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                    label="Centroids",
                )

                for i, (centroid, label) in enumerate(zip(centroids, bin_labels)):
                    ax.annotate(
                        label,
                        xy=centroid,
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        zorder=6,
                    )

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"{dim} with percentile centroids")
                ax.set_aspect("equal")
                ax.yaxis.set_inverted(True)
                #plt.colorbar(scatter, ax=ax, label="value")
        
            # if plot_stuff:
            #     # Hide unused subplot

            #     plt.tight_layout()
            
            
        # if plot_stuff:
        #     plt.show()
        if value in slices_to_save:
            if savefig_path is not None:
                fig.savefig(savefig_path.parent.joinpath(savefig_path.stem+"_"+str(value)+savefig_path.suffix), dpi = 900)

    return pd.DataFrame.from_records(all_centroid_records)




def run_gradient_analysis(
    list_to_run,
    layer_to_use=None,
    save_prefix=None,
    possible_genes=None,
    force_run_pca = False, just_do_one = False,
    flip_initial_line = False,invert_marmoset_y = False,
    selected_groups = [],selected_group_name = "",
    close_figs=True, n_gene_plot_rows = 6, n_gene_plot_cols = 3,
    n_pcs_to_keep = 10, main_figure_panels = [],
    supp_figure_panels=[]
):
    
    species_slice_ads={}
    if selected_group_name == "":
        raise ValueError("must supply name string for selected groups")
    
    if save_prefix is None:
        save_prefix = "/scratch/figure_4_panels/"+ layer_to_use + "_" + selected_group_name+"_"


    if flip_initial_line:
        for info in list_to_run:
            info["line"] = [[info["line"][0][1] ,-info["line"][0][0]], [info["line"][0][1],-info["line"][0][0]]]
        
    if possible_genes is None:
        possible_genes = {
            "Marmoset": ["CNR1", "GDA", "TESPA1", "PROX1"],
            "Macaque": ["CNR1", 'GDA', "TESPA1", "PROX1"],
            "Human": ["CNR1", 'GDA', "TESPA1","PROX1"],
        }


    if len(main_figure_panels) >0 or  len(supp_figure_panels) >0 :
        check_figure_stems = True
    else:
        check_figure_stems = False

    n_gene_plots = n_gene_plot_cols * n_gene_plot_rows

    species_gx_pc_loadings = []
    gene_sets = []
    species_best_pcs ={}
    for jj,species_ad_info in enumerate(list_to_run):

        if just_do_one & jj>0:
            break
        if force_run_pca:
            run_pca_on_spatial_cells(species_ad_info["species_ad"], pca_n_comps=n_pcs_to_keep, layer=layer_to_use, normalize=False, scale=True)
            add_pcs_to_obs(species_ad_info["species_ad"], list(range(n_pcs_to_keep)))

        
        
        species = species_ad_info["species"]
        print(species)

        gene_info_dict = dict(species=species)

        # Basic gradient analysis
        # subset to cell types of interest. this is for flexibility but in this case just makes another copy    
        strd_gradient_ad = species_ad_info["species_ad"][species_ad_info["species_ad"].obs.Group.isin(selected_groups), :].copy()
        # gradient line
        p1 = np.array(species_ad_info["line"][0])
        p2 = np.array(species_ad_info["line"][1])

        strd_gradient_ad.obs['proj_lengths_manual'] = project_2d_points_to_line(strd_gradient_ad.obs.loc[:, species_ad_info["coord_cols"]].values, p1, p2, zero_min_depth=True)

        best_pc_manual = find_best_gradient_pc(strd_gradient_ad.obsm['X_pca'], strd_gradient_ad.obs['proj_lengths_manual'])

        pcs_to_add = list(range(n_pcs_to_keep))
        # if not (best_pc in pcs_to_add):
        #     print("missing best pc " + str(best_pc) + " in first " + str(N_PCs_TO_KEEP) + " PCs for " + species + "!!")
        #     pcs_to_add.append(best_pc)
        
        #add_pcs_to_obs(strd_gradient_ad, pcs_to_add)

        # now find the best PC computationally:
        PERCENTILE_BINS = 5
        pc_spread_info = get_pc_spatial_spread(
            strd_gradient_ad.obs,
            multi_section_column="z_order",
            multi_section_values=species_ad_info["z_choices"],
            spatial_obs_columns=species_ad_info["coord_cols"],
            percentile_bins=PERCENTILE_BINS,
            plot_stuff=False,
        )


        #TODO fix pc spread info so it puts out the mean for each PC (even when all cells are being processed are used)
        # then extract top 10 best genes
        pc_means = pc_spread_info.groupby("pc").first_last_distance.mean()
        max_pc_index = pc_means.argmax()
        best_pc = int(pc_means.index[max_pc_index].strip("PC_"))
        # store for later use...
        strd_gradient_ad.uns["best_pc"] = best_pc
        species_best_pcs[species] = best_pc
        # this matches the best gradient PC (chosen by alignment to the manual line)
        
        print(best_pc, best_pc_manual, best_pc == best_pc_manual)

        plt.figure(figsize=[10, 5])
        sns.boxplot(pc_spread_info, x="pc", y="first_last_distance")
        plt.title(species+ " gradient score")
        this_save_path = pathlib.Path(save_prefix + species + "_PC_gradient_score.png")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
            plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')

        if close_figs:
            plt.close()




        # also note the explained variance
        plt.figure()
        plt.plot(species_ad_info["species_ad"].uns["pca"]["variance_ratio"], '.-',label =species)
        plt.title("explained variance ratio")
        plt.legend()
        plt.yscale("log")
        plt.xlabel("PC")
        plt.ylabel("explained variance ratio")

        this_save_path = pathlib.Path(save_prefix+ species +"_explained_variance.png")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
             plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')


        if close_figs:
            plt.close()
        # and the product of explained variance with the distance score:
        
        plt.figure(figsize=[10, 5])
        n_to_score = pc_means.shape[0]
        distance_variance_score = species_ad_info["species_ad"].uns["pca"]["variance_ratio"][:n_to_score]*pc_means
        plt.plot(range(n_to_score), distance_variance_score)
        plt.title(species+" variance-weighted gradient score")
        plt.xlabel("PC")
        plt.ylabel("variance ratio * gradient score")
        this_save_path = pathlib.Path(save_prefix + species + "_PC_gradient_score_w_variance.png")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
             plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')



        if close_figs:
            plt.close()        
        
        # if species=="Human":
        #     sign_switch =-1
        # else:
        sign_switch = 1

        
        # subset to z
        section_gradient_ad = strd_gradient_ad[strd_gradient_ad.obs['z_order'] == species_ad_info["z_choice"], :].copy()

        
        # now extract the gradient of this computationally chosen PC in this section:
   
        gradient_vector = (sign_switch*estimate_global_gradient(section_gradient_ad.obs.loc[:,species_ad_info["coord_cols"]].values,
                                                                section_gradient_ad.obs.loc[:,["PC_"+str(best_pc)]].values))
        gradient_vector = species_ad_info["arrow_size"]*gradient_vector/np.linalg.norm(gradient_vector)
        gradient_vector = np.array(gradient_vector.flat)
        print(gradient_vector)
        center_xy = section_gradient_ad.obs.loc[:, species_ad_info["coord_cols"]].mean(axis=0).values
        



        pseudo_points = [center_xy-gradient_vector/2., center_xy+gradient_vector/2.]
        print(pseudo_points)
        section_gradient_ad.obs['proj_lengths'] = project_2d_points_to_line(section_gradient_ad.obs.loc[:, species_ad_info["coord_cols"]].values, 
                                                                                pseudo_points[0], pseudo_points[1], zero_min_depth=True)
        
        
        
        
        
        
        ## plot single PC with line
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_cells_and_line(section_gradient_ad, np.array(pseudo_points), species_ad_info["coord_cols"][0], species_ad_info["coord_cols"][1], c=section_gradient_ad.obs['proj_lengths'], cmap=sns.cubehelix_palette(as_cmap=True), ax=ax, s=1)
        ax.invert_yaxis()
        futils.add_scalebar(ax, sizex=2000, sizey=0, matchx=False, matchy=False, hidex=True, hidey=True, loc='lower left')
        
        
        this_save_path = pathlib.Path(save_prefix + species + "_cells_line.png")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
             plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')
        
        this_save_path = pathlib.Path(save_prefix + species + "_cells_line.svg")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
             plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')

       
        if close_figs:
            plt.close()
        # strd_gx.plot_top_genes_spatial(n_genes=18, s=1, legend = False, palette="YlGnBu", hue_norm_max_percentile = 99)
        gene_info_dict["top_genes_pc_best"] = get_top_pc_genes(strd_gradient_ad, best_pc, top_n=n_gene_plot_rows*3, kind='both')

        gene_sets.append(gene_info_dict)

        fig, axs = plt.subplots(int(n_pcs_to_keep / 2), 2, figsize=(5, 20))
        for ii, ax in enumerate(axs.flat):

            vlim_mac_pc1 = np.nanmax(np.abs(strd_gradient_ad.obs[f'PC_{best_pc}']))

            g = plot_pc_on_single_plane(section_gradient_ad, spatial_keys=species_ad_info["coord_cols"], 
                                        pc_index=pcs_to_add[ii], vlim_abs=5, marker_size=species_ad_info["cell_marker_size"], cmap=sns.color_palette('PiYG', as_cmap=True), ax=ax)
            cbar_ax = futils.inset_axes(ax, width="3%", height="30%", loc='lower left', bbox_to_anchor=(0.90, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            cbar1 = fig.colorbar(g, cax=cbar_ax, orientation='vertical')

            cbar1.ax.set_ylabel("PC" + str(pcs_to_add[ii]), rotation=270, va='center', ha='center')

            if pcs_to_add[ii] == best_pc:
                ax.set_title("gradient PC")
            futils.add_scalebar(ax, sizex=2000, sizey=0, matchx=False, matchy=False, hidex=True, hidey=True, loc='lower left')

        plt.tight_layout()
        this_save_path = pathlib.Path(save_prefix + species + "_PCs_plot.png")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
             plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')
        
        this_save_path = pathlib.Path(save_prefix + species + "_PCs_plot.svg")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path)

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path)
        else:
             plt.savefig(this_save_path)
        
               



        if close_figs:
            plt.close()
        ## save a version with just the best PC

        fig, ax = plt.subplots(figsize=(5, 5))
        g = plot_pc_on_single_plane(section_gradient_ad, spatial_keys=species_ad_info["coord_cols"],
                                    pc_index=best_pc, vlim_abs=5, marker_size=species_ad_info["cell_marker_size"], cmap=sns.color_palette('PiYG', as_cmap=True), ax=ax)
        cbar_ax = futils.inset_axes(ax, width="3%", height="30%", loc='lower left', bbox_to_anchor=(0.90, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar1 = fig.colorbar(g, cax=cbar_ax, orientation='vertical')

        cbar1.ax.set_ylabel("PC" + str(best_pc), rotation=270, va='center', ha='center')
        # if species == "Marmoset":
        #     ax.invert_yaxis()

        # if species == "Human":
        #     arrow_sign = -1
        # else:
        arrow_sign = 1

        add_pc_gradient_arrow_to_section(
            ax,
            section_gradient_ad.obs.loc[:, species_ad_info["coord_cols"]].values,
            section_gradient_ad.obs.loc[:, "PC_" + str(best_pc)].values.ravel(),
            arrow_scale=arrow_sign * species_ad_info["arrow_size"] / 10.,
            arrow_color='black',
            arrow_linewidth=2,
            arrow_alpha=1,
        )



        this_save_path = pathlib.Path(save_prefix + species + "_one_PC_plot.png")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
             plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')
        

        this_save_path = pathlib.Path(save_prefix + species + "_one_PC_plot.svg")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900)

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900)
        else:
             plt.savefig(this_save_path, dpi = 900)
        



        if close_figs:
            plt.close()        
        # plot top genes from PCg
        
        hue_norm_max_percentile = 99
        fig, axs = plt.subplots(n_gene_plot_rows, n_gene_plot_cols, figsize=(8*n_gene_plot_cols/3., n_gene_plot_rows*3))

        for ii, ax in enumerate(axs.flat):
            gene_to_plot = gene_info_dict["top_genes_pc_best"][ii]
            hn = (0, np.percentile(section_gradient_ad.obs_vector(gene_to_plot, layer=layer_to_use), hue_norm_max_percentile))

            gr = futils.plot_spatial_gene_expression(section_gradient_ad, gene_to_plot, species_ad_info["coord_cols"][0], 
                                              species_ad_info["coord_cols"][1], None, None, ax, layer=layer_to_use, sort_values=True, 
                                              use_seaborn=True, hue_norm=hn,edgecolors="none",
                                              s=species_ad_info["cell_marker_size"], palette='YlGnBu', legend=False)
            ax.set_title(gene_to_plot)
            futils.add_scalebar(ax, sizex=2000, sizey=0, matchx=False, matchy=False, hidex=True, hidey=True, loc='lower left')
            print(gene_to_plot, np.mean(section_gradient_ad[:, gene_to_plot].X))
            ax.invert_yaxis()
        plt.tight_layout()



        this_save_path = pathlib.Path(save_prefix + species + "_"+str(n_gene_plots)+"_geneplot.png")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
             plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')
        
        this_save_path = pathlib.Path(save_prefix + species + "_"+str(n_gene_plots)+"_geneplot.svg")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path)

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path)
        else:
             plt.savefig(this_save_path)
        





        if close_figs:
            plt.close()
        fig, axs = plt.subplots(2, 2, figsize=(5, 5))

        for ii, ax in enumerate(axs.flat):
            gene_to_plot = possible_genes[species][ii]
            hn = (0, np.percentile(section_gradient_ad.obs_vector(gene_to_plot, layer=layer_to_use), hue_norm_max_percentile))

            gr = futils.plot_spatial_gene_expression(section_gradient_ad, gene_to_plot, species_ad_info["coord_cols"][0],
                                              species_ad_info["coord_cols"][1], None, None, ax, layer=layer_to_use, sort_values=True, 
                                              use_seaborn=True, hue_norm=hn,edgecolors="none",
                                              s=species_ad_info["cell_marker_size"], palette='YlGnBu', legend=False)

            ax.set_title(gene_to_plot)
            ax.invert_yaxis()

            futils.add_scalebar(ax, sizex=2000, sizey=0, matchx=False, matchy=False, hidex=True, hidey=True, loc='lower left')
        plt.tight_layout()
        plt.savefig(save_prefix + species + "_4_geneplot.png", dpi=900)
        plt.savefig(save_prefix + species + "_4_geneplot.svg")


        this_save_path = pathlib.Path(save_prefix + species + "_4_geneplot.png")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
             plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')

        this_save_path = pathlib.Path(save_prefix + species + "_4_geneplot.svg")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path)

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path)
        else:
             plt.savefig(this_save_path)
        
        



        if close_figs:
            plt.close()
        

        fig, axs = plt.subplots(int(np.ceil(len(selected_groups)/2)), 2, figsize=(8, 8))
        for ii, group_to_plot in enumerate(selected_groups):
            group_sub = section_gradient_ad[section_gradient_ad.obs.Group == group_to_plot, :]

            ax = axs.flat[ii]
            g = plot_pc_on_single_plane(group_sub, spatial_keys=species_ad_info["coord_cols"],
                                        pc_index=best_pc, vlim_abs=5, marker_size=species_ad_info["cell_marker_size"], cmap=sns.color_palette('managua', as_cmap=True), ax=ax)
            # cbar_ax = inset_axes(ax, width="3%", height="30%", loc='lower left', bbox_to_anchor=(0.90, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            # cbar1 = fig.colorbar(g, cax=cbar_ax, orientation='vertical')

            # cbar1.ax.set_ylabel("PC" + str(best_pc), rotation=270, va='center', ha='center')
            #ax.invert_yaxis()
            if pcs_to_add[ii] == best_pc:
                ax.set_title("gradient PC")
            futils.add_scalebar(ax, sizex=2000, sizey=0, matchx=False, matchy=False, hidex=True, hidey=True, loc='lower left')

            ax.set_title(group_to_plot)
        plt.tight_layout()
        plt.savefig(save_prefix + species + "_PC_in_SPN_groups.png", dpi=900)
        plt.savefig(save_prefix + species + "_PC_in_SPN_groups.svg")

        this_save_path = pathlib.Path(save_prefix + species + "_PC_in_SPN_groups.png")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path, dpi=900,bbox_inches='tight')
        else:
             plt.savefig(this_save_path, dpi = 900,bbox_inches='tight')
        

        this_save_path = pathlib.Path(save_prefix + species + "_PC_in_SPN_groups.svg")
        if check_figure_stems:
            if this_save_path.stem in main_figure_panels:
                save_path = this_save_path.parent.joinpath("FIGURE_4_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path)

            if this_save_path.stem in supp_figure_panels:
                save_path = this_save_path.parent.joinpath("SUPP_"+this_save_path.stem+this_save_path.suffix)
                plt.savefig(save_path)
        else:
             plt.savefig(this_save_path)
        





        if close_figs:
            plt.close()
        
        species_slice_ads[species] = section_gradient_ad
    return dict(gene_sets = list(gene_sets), species_slice_ads=species_slice_ads, species_best_pcs=species_best_pcs)