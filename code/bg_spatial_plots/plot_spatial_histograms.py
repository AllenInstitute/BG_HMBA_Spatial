import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import nibabel as nib


def calculate_2D_histogram(
    adata_sec, 
    coords_obsm_key='spatial_grid',
    x_col=None,
    y_col=None, 
    binsize=300,
    obs_field=None, 
    category=None, 
    normalization=None,
    reference_adata=None,
):
    """
    Calculate a 2D histogram of counts for a specific category in an .obs field,
    with control over the binsize in physical units (e.g., pixels or micrometers),
    and optional normalization to the total number of cells.
    
    *Assumes you have a single section in your AnnData object.*

    Parameters
    ----------
    adata: AnnData object
    coords_obsm_key: str
        The key in adata.obsm containing 2D spatial coordinates
    binsize: float
        Size of each bin in the same units as the specified spatial coordinates
    obs_field: str
        The field in adata.obs to filter by
    category: str
        The category within the obs_field to calculate the histogram for
    normalize: str, 
        Normalization method; options include:
        - 'proportion': Normalize each bin by the total number of cells in that
                        bin in the full anndata object. NB: If you want to 
                        restrict the denominator to just a specific subset of 
                        categories, you'll need to filter the anndata object 
                        before passing it into this function. e.g. if you 
                        want your denominator to be just Astrocytes, pass in 
                        adata[adata.obs['Group']=='Astrocytes'].
        - 'density': Normalize by the area of each bin.
        - 'proportion_and_density': Normalize by both the total number of 
                                    cells and the area of each bin.
        - None: No normalization.
    reference_adata: AnnData object, optional
        If provided, bins with no cells in this reference dataset will be set to NaN
        (displayed as background color). This allows you to filter adata_sec for 
        calculation but still show the spatial extent of the full dataset.

    Returns:
    - H: 2D histogram of counts (raw or normalized, with NaN for bins with no cells)
    - xedges, yedges: Bin edges
    """
    
    # # Get spatial coordinates
    # if x_col is not None and y_col is not None:
    #     coords = adata_sec.obs[[x_col, y_col]].values
    # else:
    #     if coords_obsm_key not in adata_sec.obsm:
    #         raise ValueError(f"Coordinates key '{coords_obsm_key}' not found in adata.obsm.")
    #     coords = adata_sec.obsm[coords_obsm_key]

    # # Determine the range of the spatial coordinates
    # x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    # y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # # Calculate the number of bins based on the binsize
    # x_bins = int((x_max - x_min) / binsize)
    # y_bins = int((y_max - y_min) / binsize)
    # # print(f"Calculating histogram with {x_bins} x {y_bins} bins of size {binsize} um")
    
    # # Calculate the 2D histogram for all cells in the anndata object
    # H_all, xedges_all, yedges_all = np.histogram2d(
    #     coords[:, 0], coords[:, 1],
    #     bins=[x_bins, y_bins],
    #     range=[[x_min, x_max], [y_min, y_max]]
    # )

    # # If category is specified, only calculate the histogram for that category
    # if obs_field is not None and category is not None:
    #     # user input checks
    #     if obs_field not in adata_sec.obs.columns:
    #         raise ValueError(f"Field '{obs_field}' not found in adata.obs.")
    #     if category not in adata_sec.obs[obs_field].cat.categories:
    #         print(f"Warning: Category '{category}' not found in field '{obs_field}'.")

    #     # Filter coordinates based on the specified category
    #     mask = adata_sec.obs[obs_field] == category
    #     coords_filtered = coords[mask]

    #     # Calculate the 2D histogram for the specified category
    #     H_category, xedges_category, yedges_category = np.histogram2d(
    #         coords_filtered[:, 0], coords_filtered[:, 1],
    #         bins=[x_bins, y_bins],
    #         range=[[x_min, x_max], [y_min, y_max]]
    #     )

    #     # Set bins with no cells at all to NaN (to distinguish from bins with no cells just in this category)
    #     H_category[H_all == 0] = np.nan

    #     # If normalization is specified, apply it:
    #     if normalization == 'proportion':
    #         # Normalize the histogram by the total counts
    #         H_category = H_category / (H_all + 1e-6)  # Add a small epsilon to avoid division by zero
    #     elif normalization == 'density':
    #         # Normalize by just area of each bin
    #         bin_area = binsize ** 2
    #         H_category = H_category / bin_area
    #     elif normalization == 'proportion_and_density':
    #         # Normalize by the total number of cells AND the area of each bin
    #         H_category = H_category / (H_all + 1e-6) / (binsize ** 2)

    #     return H_category, xedges_category, yedges_category
    # # If no category is specified, simply return the histogram for all cells
    # else:
    #     return H_all, xedges_all, yedges_all

    # Determine which dataset to use for spatial extent and "no cells" masking
    spatial_reference = reference_adata if reference_adata is not None else adata_sec
    
    # Get spatial coordinates from the reference dataset for range calculation
    if x_col is not None and y_col is not None:
        coords_ref = spatial_reference.obs[[x_col, y_col]].values
    else:
        if coords_obsm_key not in spatial_reference.obsm:
            raise ValueError(f"Coordinates key '{coords_obsm_key}' not found in reference adata.obsm.")
        coords_ref = spatial_reference.obsm[coords_obsm_key]

    # Get spatial coordinates from the actual data for calculation
    if x_col is not None and y_col is not None:
        coords = adata_sec.obs[[x_col, y_col]].values
    else:
        if coords_obsm_key not in adata_sec.obsm:
            raise ValueError(f"Coordinates key '{coords_obsm_key}' not found in adata.obsm.")
        coords = adata_sec.obsm[coords_obsm_key]

    # Determine the range based on the reference dataset
    x_min, x_max = coords_ref[:, 0].min(), coords_ref[:, 0].max()
    y_min, y_max = coords_ref[:, 1].min(), coords_ref[:, 1].max()

    # Calculate the number of bins based on the binsize
    x_bins = int((x_max - x_min) / binsize)
    y_bins = int((y_max - y_min) / binsize)
    
    # Calculate the 2D histogram for all cells in the reference dataset (for masking)
    H_ref_all, xedges_all, yedges_all = np.histogram2d(
        coords_ref[:, 0], coords_ref[:, 1],
        bins=[x_bins, y_bins],
        range=[[x_min, x_max], [y_min, y_max]]
    )

    # Calculate the 2D histogram for all cells in the actual dataset (for normalization denominator)
    H_all, _, _ = np.histogram2d(
        coords[:, 0], coords[:, 1],
        bins=[x_bins, y_bins],
        range=[[x_min, x_max], [y_min, y_max]]
    )

    # If category is specified, only calculate the histogram for that category
    if obs_field is not None and category is not None:
        # user input checks
        if obs_field not in adata_sec.obs.columns:
            raise ValueError(f"Field '{obs_field}' not found in adata.obs.")
        if category not in adata_sec.obs[obs_field].cat.categories:
            print(f"Warning: Category '{category}' not found in field '{obs_field}'.")

        # Filter coordinates based on the specified category
        mask = adata_sec.obs[obs_field] == category
        coords_filtered = coords[mask]

        # Calculate the 2D histogram for the specified category
        H_category, xedges_category, yedges_category = np.histogram2d(
            coords_filtered[:, 0], coords_filtered[:, 1],
            bins=[x_bins, y_bins],
            range=[[x_min, x_max], [y_min, y_max]]
        )

        # Set bins with no cells in the REFERENCE dataset to NaN
        H_category[H_ref_all == 0] = np.nan

        # If normalization is specified, apply it:
        if normalization == 'proportion':
            # Normalize the histogram by the total counts in the actual dataset
            H_category = H_category / (H_all + 1e-6)  # Add a small epsilon to avoid division by zero
        elif normalization == 'density':
            # Normalize by just area of each bin
            bin_area = binsize ** 2
            H_category = H_category / bin_area
        elif normalization == 'proportion_and_density':
            # Normalize by the total number of cells AND the area of each bin
            H_category = H_category / (H_all + 1e-6) / (binsize ** 2)

        return H_category, xedges_category, yedges_category
    # If no category is specified, simply return the histogram for all cells
    else:
        # Set bins with no cells in the reference dataset to NaN
        H_all[H_ref_all == 0] = np.nan
        return H_all, xedges_all, yedges_all


def plot_2D_histogram(
    H, 
    xedges, 
    yedges,
    cmap='viridis', 
    vlims=(0, None), 
    imshow_size=None,
    no_cells_color='lightgrey', 
    no_cells_in_category_color='black', 
    highlight_zero = False,
    ax=None,
    title=None,
    show_legend=True,
    normalization=None,  # sets colorbar label units
    show_cbar=True,
    return_im=False
):
    """
    Plots a 2D histogram with custom colormap handling.

    Parameters
    ----------
    H : 2D array
        The histogram data to plot.
    xedges : array
        The bin edges along the x-axis.
    yedges : array
        The bin edges along the y-axis.
    cmap : str, optional
        The colormap to use for the plot. Default is 'viridis'.
    vlims : tuple, optional
        The (vmin, vmax) values for normalization. Default is (0, 1).
    no_cells_color : str, optional
        Color for bins with no cells. Default is 'white'.
    no_category_color : str, optional
        Color for bins with cells but none in this category. Default is 'gray'.
    ax : matplotlib axis, optional
        The axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    im : AxesImage
    """

    # Build custom colormap
    custom_cmap = plt.get_cmap(cmap)
    custom_cmap.set_bad(no_cells_color)  # Set color for bins with no cells
    custom_cmap.set_under(no_cells_in_category_color)  # Set color for bins with cells but none in this category

    # Dynamically set the extent
    if imshow_size is None:
        imshow_extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
        H_padded = H  # No padding needed if imshow_size is None
    else:
        binsize = int(xedges[1] - xedges[0])  # Assuming uniform bin size
        H_padded, imshow_extent = pad_histogram_to_physical_plot_size(H, binsize, imshow_size, pad_value=np.nan)

    # set vlims
    if vlims[1] is None:
        if normalization=='proportion':
            vlims = (0, 1.0)
        else:
            vlims = (vlims[0], np.nanmax(H))

    # Normalize
    norm = mcolors.Normalize(vmin=vlims[0], vmax=vlims[1], clip=False)

    if highlight_zero:
        # set zero to slightly negative value so it can be colored with the under color
        H_padded = np.where(H_padded == 0, -1e-10, H_padded)  # Set zero counts to a small negative value to trigger cmap.set_under()

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    im = ax.imshow(
        H_padded.T,  # transpose for orientation
        origin='lower',
        cmap=custom_cmap,
        norm=norm,
        extent=imshow_extent,
    )

    if show_cbar:
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        if normalization == 'proportion':
            cbar_label = 'Proportion of Total Cells'
        elif normalization == 'density':
            cbar_label = 'Density (cells per um²)'
        elif normalization == 'proportion_and_density':
            cbar_label = 'Proportion and Density (fraction per um²)'
        elif normalization=='enrichment':
            cbar_label = 'log2(Enrichment)'
        else:
            cbar_label = 'Counts'
        cbar.set_label(cbar_label)
    
    # Add legend patches
    if show_legend:
        legend_patches = [
            mpatches.Patch(color=no_cells_color, label='No spatial cells'),
            mpatches.Patch(color=no_cells_in_category_color, label='Zero cells in category'),
        ]
        ax.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 1.0))
    if title is not None:
        ax.set_title(title, pad=50)
    ax.axis('off')

    if return_im:
        return im
    else:
        return ax


def pad_histogram_to_physical_plot_size(H, binsize_um, plot_size_um, pad_value=np.nan):
    """
    Pads a 2D histogram array H so it fills a square plot of size plot_size_um x plot_size_um,
    with original data centered as best as possible.

    Parameters:
    - H: 2D numpy array
    - binsize_um: physical size of each bin (e.g., 300)
    - plot_size_um: desired physical size of plot in microns (e.g., 300*120)
    - pad_value: value to pad with (default: np.nan)

    Returns:
    - H_padded: padded 2D array
    - extent: list for imshow extent: [0, plot_size_um, 0, plot_size_um]
    """
    h, w = H.shape
    target_bins = plot_size_um // binsize_um
    if h > target_bins or w > target_bins:
        raise ValueError(f"H shape ({h}, {w}) exceeds target_bins ({target_bins})")

    pad_top = (target_bins - h) // 2
    pad_bottom = target_bins - h - pad_top
    pad_left = (target_bins - w) // 2
    pad_right = target_bins - w - pad_left

    H_padded = np.pad(
        H,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=pad_value
    )

    extent = [0, plot_size_um, 0, plot_size_um]
    return H_padded, extent