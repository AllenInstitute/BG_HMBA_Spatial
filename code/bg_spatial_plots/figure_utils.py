import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib as mpl
import seaborn as sns
from bg_spatial_plots.constants_color_palettes import CURRENT_COLOR_DICT_OF_DICTS
import bg_spatial_plots.color_utils as cutils
import numpy as np
import shapely
from shapely.plotting import plot_polygon
from matplotlib import font_manager

def add_scalebar(
    ax, 
    matchx=True, matchy=True, 
    hidex=True, hidey=True, 
    loc='lower right', 
    fontsize=10, 
    barcolor='black', barwidth=2, 
    offset=(0.05, 0.05), 
    label_spacing=0.015,
    **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - loc : location ('lower right', 'lower left', 'upper right', 'upper left')
    - fontsize : font size for labels
    - barcolor : color of bars and text
    - barwidth : width of bars
    - offset : offset from corner as fraction of axes size
    - **kwargs : can include sizex, sizey, labelx, labely for manual specification
    Returns None
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    sizex = 0
    sizey = 0
    labelx = None
    labely = None
    
    if matchx:
        sizex = f(ax.xaxis)
        labelx = str(sizex) if sizex else None
    else:
        sizex = kwargs.get('sizex', 0)
        labelx = kwargs.get('labelx', None)
        
    if matchy:
        sizey = f(ax.yaxis)
        labely = str(sizey) if sizey else None
    else:
        sizey = kwargs.get('sizey', 0)
        labely = kwargs.get('labely', None)
    
    # Use simple matplotlib elements
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate base position based on location
    if 'right' in loc:
        x_base = xlim[1] - (xlim[1] - xlim[0]) * offset[0]
    else:  # left
        x_base = xlim[0] + (xlim[1] - xlim[0]) * offset[0]
    
    if 'lower' in loc:
        y_base = ylim[0] + (ylim[1] - ylim[0]) * offset[1]
    else:  # upper
        y_base = ylim[1] - (ylim[1] - ylim[0]) * offset[1]
    
    # Adjust positions based on what bars we're drawing
    if sizex and sizey:
        # Both bars - position them in an L shape
        if 'right' in loc:
            x_pos = x_base - sizex
        else:
            x_pos = x_base
            
        if 'lower' in loc:
            y_pos = y_base
            y_vert_pos = y_base
        else:
            y_pos = y_base
            y_vert_pos = y_base - sizey
            
    elif sizex:
        # Only horizontal bar
        if 'right' in loc:
            x_pos = x_base - sizex
        else:
            x_pos = x_base
        y_pos = y_base
        
    elif sizey:
        # Only vertical bar
        x_pos = x_base
        if 'lower' in loc:
            y_vert_pos = y_base
        else:
            y_vert_pos = y_base - sizey
    
    # Draw horizontal bar if requested
    if sizex:
        ax.plot([x_pos, x_pos + sizex], [y_pos, y_pos], 
                color=barcolor, linewidth=barwidth, solid_capstyle='butt')
        
        if labelx:
            # Position label below the bar
            label_y = y_pos - (ylim[1] - ylim[0]) * label_spacing
            ax.text(x_pos + sizex/2, label_y, labelx, 
                   ha='center', va='top', fontsize=fontsize, color=barcolor)
    
    # Draw vertical bar if requested
    if sizey:
        ax.plot([x_pos, x_pos], [y_vert_pos, y_vert_pos + sizey], 
                color=barcolor, linewidth=barwidth, solid_capstyle='butt')
        
        if labely:
            # Position label to the left of the bar
            label_x = x_pos - (xlim[1] - xlim[0]) * label_spacing
            ax.text(label_x, y_vert_pos + sizey/2, labely, 
                   ha='right', va='center', fontsize=fontsize, color=barcolor, rotation=90)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)
    if hidex and hidey: ax.set_frame_on(False)

def plot_spatial_subset(
    data, 
    xcol, ycol, 
    zcol_name, zcol_val, 
    fg_col, fg_vals, fg_kwargs, 
    bg_kwargs, 
    ax,
    palette=None,
    show_fg=True
):
    """Plot single plane of spatial data, with only some cells in the foreground or just the background."""
    subset = data[data[zcol_name] == zcol_val]
    bg = subset if not show_fg else subset[~subset[fg_col].isin(fg_vals)]
    
    # Plot background
    ax.scatter(bg[xcol], bg[ycol], **bg_kwargs)
    
    if show_fg and fg_col and fg_vals:
        fg = subset[subset[fg_col].isin(fg_vals)]
        point_palette = cutils.generate_color_palette(fg[fg_col].unique().tolist(), palette)
        ax.scatter(fg[xcol], fg[ycol], c=fg[fg_col].map(point_palette), **fg_kwargs)
    
    format_spatial_axes(ax)
    set_square_lims(ax, subset[xcol], subset[ycol])


def format_spatial_axes(ax, invert_y=False, show_spines=False, show_axes_labels_ticks=False):
    # always do these
    ax.set_aspect('equal')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # may need to invert yaxis for certain kinds of data
    if invert_y:
        ax.invert_yaxis()

    # optionally hide the spines, and/or the axis ticks & tick labels
    if not show_spines:
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    if not show_axes_labels_ticks:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def set_square_lims(ax, x_data, y_data, buffer=0.05):
    """
    Set axis limits to a square bounding box around the data.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify
    x_data : array-like
        X coordinates of the data
    y_data : array-like  
        Y coordinates of the data
    buffer : float, default=0.05
        Buffer around the data as a fraction of the range (0.05 = 5%)
        
    Returns
    -------
    None
        Modifies the axes in place
    """
    # Get data ranges
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    
    # Calculate ranges
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Use the larger range to make it square
    max_range = max(x_range, y_range)
    
    # Calculate centers
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Calculate buffer in data units
    buffer_size = max_range * buffer
    
    # Set square limits centered on data
    half_range = max_range / 2 + buffer_size
    
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)

def highlight_clusters_cmap(all_clusters, select_clusters, colors, default_color='lightgrey'):
    """
    Generate a colormap where select clusters are colored by cutom colors.
    The rest will be set to a default color.
    """
    cluster_cmap = {}
    count = 0
    for cluster in all_clusters:
        if cluster in select_clusters:
            color = colors[count]
            cluster_cmap.update({cluster: color})
            count += 1
        else:
            cluster_cmap.update({cluster: default_color})
    
    return cluster_cmap

def add_fonts_to_manager():
    # locate current conda environment's font folder
    conda_env_path = os.environ.get("CONDA_PREFIX")
    if conda_env_path:
        conda_font_dir = os.path.join(conda_env_path, "fonts")
        
        # find and register every TTF file in that directory
        if os.path.exists(conda_font_dir):
            for font in font_manager.findSystemFonts(fontpaths=[conda_font_dir], fontext='ttf'):
                font_manager.fontManager.addfont(font)

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    # embed as TrueType (not Type 3) so font is actual editable text, 
    # not paths/outlines, in vector formats
    plt.rcParams['pdf.fonttype'] = 42   
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

def plot_staggered_slab_sections(
    slab_tables,
    hemisphere_outlines=None,
    gene_names=['DRD1'],
    figsize=(12, 8),
    offset_x=0.15,
    offset_y=0.1,
    cmap="Grays",
    mask_size=500,
    spacing_mode="per_slice",     # "per_slice" | "global" | "fixed"
    global_percentile=100         # only for "global": 100=max, 75/50 for smaller steps
):
    """
    Create a staggered visualization of slab sections with optional hemisphere outlines.

    Parameters
    ----------
    slab_tables : list
        List of slab_spot_table objects.
    hemisphere_outlines : list of ndarray, optional
        List of Nx2 arrays with polygon coordinates for each section.
    gene_names : list
        List of gene names to visualize.
    figsize : tuple
        Figure size.
    offset_x, offset_y : float
        Offset amounts for staggering effect.
    cmap : str
        Colormap for the heatmaps.
    mask_size : int
        Minimum size of zero regions to mask.
    spacing_mode : str
        "per_slice" -> original behavior (offset scales by each slice's effective size).
        "global"    -> uniform step using a global effective size (see global_percentile).
        "fixed"     -> constant step: i*offset_x, i*offset_y (ignores sizes).
    global_percentile : int
        Percentile (0–100) of effective sizes used for the global step (100=max).
    """

    import numpy as np
    from scipy import ndimage
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    image_data = []
    original_extents = []

    # ---------- FIRST PASS: Determine global extents including polygons ----------
    all_extents = []
    for i, slab_spot_table in enumerate(slab_tables):
        temp_fig, temp_ax = plt.subplots(figsize=(6, 6))
        slab_spot_table.get_genes(gene_names=gene_names).show_binned_heatmap(ax=temp_ax)

        # Include polygon so global bounds account for it
        if hemisphere_outlines is not None:
            hem_coords = hemisphere_outlines[i]
            temp_ax.plot(hem_coords[:, 0], hem_coords[:, 1], color='none')

        extent = temp_ax.get_xlim() + temp_ax.get_ylim()
        all_extents.append(extent)
        plt.close(temp_fig)

    min_x = min(e[0] for e in all_extents)
    max_x = max(e[1] for e in all_extents)
    min_y = min(e[2] for e in all_extents)
    max_y = max(e[3] for e in all_extents)

    # ---------- SECOND PASS: Get image arrays & original extents ----------
    for i, slab_spot_table in enumerate(slab_tables):
        temp_fig, temp_ax = plt.subplots(figsize=(6, 6))
        slab_spot_table.get_genes(gene_names=gene_names).show_binned_heatmap(ax=temp_ax)

        temp_ax.set_xlim(min_x, max_x)
        temp_ax.set_ylim(min_y, max_y)
        temp_ax.set_aspect('equal')
        temp_fig.canvas.draw()

        img_obj = temp_ax.get_images()[0]
        img_array = img_obj.get_array()
        original_extent = img_obj.get_extent()

        image_data.append(img_array)
        original_extents.append(original_extent)
        plt.close(temp_fig)

    # ---------- Normalize dimensions ----------
    max_dimension = 0
    for extent in original_extents:
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]
        max_dimension = max(max_dimension, width, height)

    normalized_widths = []
    normalized_heights = []
    for extent in original_extents:
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]
        scale_factor = 1.0 / max_dimension
        normalized_widths.append(width * scale_factor)
        normalized_heights.append(height * scale_factor)

    # ---------- Precompute polygon widths/heights (scaled like images) ----------
    poly_widths = []
    poly_heights = []
    for i, extent in enumerate(original_extents):
        ox0, ox1, oy0, oy1 = extent
        width = normalized_widths[i]
        height = normalized_heights[i]
        if hemisphere_outlines is not None:
            hem_coords = hemisphere_outlines[i]
            scale_x = width / (ox1 - ox0)
            scale_y = height / (oy1 - oy0)
            sx = (hem_coords[:, 0] - ox0) * scale_x
            sy = (hem_coords[:, 1] - oy0) * scale_y
            poly_widths.append(sx.max() - sx.min())
            poly_heights.append(sy.max() - sy.min())
        else:
            poly_widths.append(0.0)
            poly_heights.append(0.0)

    # Effective sizes per slice (match original logic: image vs polygon)
    eff_w = np.maximum(normalized_widths, poly_widths)
    eff_h = np.maximum(normalized_heights, poly_heights)

    # Global step size (percentile of effective sizes) for "global" mode
    global_w = np.percentile(eff_w, global_percentile)
    global_h = np.percentile(eff_h, global_percentile)

    # ---------- FINAL PASS: Plot staggered images & polygons ----------
    max_x_extent = 0
    max_y_extent = 0

    for i, img_array in enumerate(reversed(image_data)):
        section_idx = len(image_data) - 1 - i

        # Scaled image dims
        width = normalized_widths[section_idx]
        height = normalized_heights[section_idx]

        # Scaled polygon coords (for drawing)
        ox0, ox1, oy0, oy1 = original_extents[section_idx]
        if hemisphere_outlines is not None:
            hem_coords = hemisphere_outlines[section_idx]
            scale_x = width / (ox1 - ox0)
            scale_y = height / (oy1 - oy0)
            scaled_coords_x = (hem_coords[:, 0] - ox0) * scale_x
            scaled_coords_y = (hem_coords[:, 1] - oy0) * scale_y
        else:
            scaled_coords_x = scaled_coords_y = np.array([])

        # ---------- Offsets by mode ----------
        if spacing_mode == "global":
            x_offset = section_idx * offset_x * global_w
            y_offset = section_idx * offset_y * global_h
        elif spacing_mode == "fixed":
            x_offset = section_idx * offset_x
            y_offset = section_idx * offset_y
        else:  # "per_slice" (original behavior)
            # Use per-slice effective sizes (image vs polygon)
            effective_width = eff_w[section_idx]
            effective_height = eff_h[section_idx]
            x_offset = section_idx * offset_x * effective_width
            y_offset = section_idx * offset_y * effective_height

        # Final extent for this image
        extent = [x_offset, x_offset + width, y_offset, y_offset + height]
        max_x_extent = max(max_x_extent, extent[1])
        max_y_extent = max(max_y_extent, extent[3])

        # ---------- Mask large zero regions ----------
        zero_mask = (img_array == 0)
        structure = np.ones((3, 3))
        labeled_zeros, num_features = ndimage.label(zero_mask, structure=structure)
        final_mask = np.zeros_like(zero_mask, dtype=bool)
        for j in range(1, num_features + 1):
            region = (labeled_zeros == j)
            if np.sum(region) > mask_size:
                final_mask |= region

        # ---------- Plot masked or full image with zorder i+1 ----------
        if np.any(final_mask):
            masked_array = np.ma.masked_where(final_mask, img_array)
            ax.imshow(masked_array, extent=extent, cmap=cmap, origin='lower', zorder=i+1)
        else:
            ax.imshow(img_array, extent=extent, cmap=cmap, origin='lower', zorder=i+1)

        # ---------- Plot scaled polygon with zorder i ----------
        if hemisphere_outlines is not None and len(scaled_coords_x) > 0:
            ax.fill(scaled_coords_x + x_offset, scaled_coords_y + y_offset,
                    edgecolor='k', linewidth=1, facecolor='w', alpha=0.5, zorder=i)

    ax.set_xlim(-0.2, max_x_extent + 0.2)
    ax.set_ylim(-0.2, max_y_extent + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    fig._max_dimension = max_dimension
    return fig, ax

def plot_section_montage(
    obs, 
    x_coord_col, 
    y_coord_col, 
    fg_kwargs, 
    bg_kwargs, 
    ncols=9, 
    h_overlap=0.01, 
    v_overlap=-0.1, 
    stagger=0.01, 
    flip_y=True, 
    zoom_radius=100, 
    z_order=None, 
    hemisphere_outlines=None, 
    scalebar=None,
    show_region_circles=True,
    extra_bg_obs=None,
    extra_bg_kwargs={},
    show=True,
    figscale=1.5
): 
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from .coordinate_transforms import get_annotation_coords
    from matplotlib.patches import Circle

    if z_order is not None:
        z_planes = z_order
    else:
        z_planes = sorted(obs['z_order'].unique().tolist())
    n_planes = len(z_planes)
    nrows = (n_planes + ncols) // ncols

    neurons = obs[obs['Neighborhood']!='Nonneuron']['Group'].unique().tolist()

    palette = sns.color_palette("tab10", 5)
    regions = ['STR', 'GPe', 'GPi', 'SN', 'STH']
    region_colors = {region: palette[i % 5] for i, region in enumerate(regions)}

    fig = plt.figure(figsize=(ncols*figscale, nrows*figscale))
    
    subplot_width = 0.8 / ncols
    subplot_height = 0.8 / nrows
    
    centered_coords = []
    for row in range(nrows):
        for col in reversed(range(ncols)):
            i = row * ncols + col
            if i < n_planes:
                z_plane= z_planes[i]
            else:
                continue
        
            row = i // ncols
            col = i % ncols

            # Calculate position with overlap and stagger
            left = 0.1 + col * (subplot_width - h_overlap)
            bottom = 0.9 - (row + 1) * (subplot_height - v_overlap)

            # Add progressive upward stagger for each column
            bottom += col * stagger

            ax = fig.add_axes([left, bottom, subplot_width, subplot_height])
            
            obs_zplane = obs[obs['z_order']==z_plane].copy()

            coords = obs_zplane[[x_coord_col, y_coord_col]].to_numpy()
            mean_coords = np.mean(coords, axis=0)
            
            if hemisphere_outlines is not None:
                try:
                    section_id = obs_zplane['slab'].unique().tolist()[0]
                    section_id = section_id.replace('.', '')
                    hem_outline_coords = get_annotation_coords(section_id, hemisphere_outlines)
                except:
                    section_id = obs_zplane['brain_section_barcode'].unique().tolist()[0]
                    hem_outline_coords = get_annotation_coords(section_id, hemisphere_outlines)
                    # bug fix because get_annotation_coords sometimes returns the array wrapped in a list
                    if isinstance(hem_outline_coords, list):
                        hem_outline_coords = hem_outline_coords[0]
                mean_coords = np.mean(hem_outline_coords, axis=0)
                
            centered = coords - mean_coords
            obs_zplane['centered_x'] = centered[:, 0]
            obs_zplane['centered_y'] = centered[:, 1]
            centered_coords.append(centered)

            # manually add extra background observations if provided
            if extra_bg_obs is not None:
                extra_bg_zplane = extra_bg_obs[extra_bg_obs['z_order']==z_plane].copy()
                if not extra_bg_zplane.empty:
                    # Apply same centering transformation
                    extra_bg_zplane['centered_x'] = extra_bg_zplane[x_coord_col] - mean_coords[0]  # Fixed
                    extra_bg_zplane['centered_y'] = extra_bg_zplane[y_coord_col] - mean_coords[1]  # Fixed
                    sns.scatterplot(extra_bg_zplane, x='centered_x', y='centered_y', 
                                    ax=ax, **extra_bg_kwargs)        
            
            fg = obs_zplane[obs_zplane['Group'].isin(neurons)]
            bg = obs_zplane[~obs_zplane['Group'].isin(neurons)]
            bg = bg.sample(frac=0.5)
            sns.scatterplot(bg, x='centered_x', y='centered_y', ax=ax, **bg_kwargs)
            sns.scatterplot(fg, x='centered_x', y='centered_y', ax=ax, **fg_kwargs)

            if hemisphere_outlines is not None:
                hem_centered_coords = hem_outline_coords - mean_coords
                ax.plot(hem_centered_coords[:, 0], hem_centered_coords[:, 1], color='grey', linewidth=1)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

            # add region circles
            if show_region_circles:
                regions_zplane = obs_zplane[regions].sum()
                regions_zplane = regions_zplane[regions_zplane > 0].index.to_list()
                regions_zplane = [r for r in regions_zplane if r in regions]
                x0, dx = 0.3, 0.11   # starting x and spacing (in axes coords)
                y0, r  = 1.03, 0.035  # slightly above axes; radius
                for j, region in enumerate(regions_zplane):
                    circ = Circle((x0 + j*dx, y0), r,
                            transform=ax.transAxes,
                            facecolor=region_colors[region],
                            clip_on=False, zorder=1000)
                    ax.add_patch(circ)


    all_points = np.vstack(centered_coords)
    x_min, x_max = min(all_points[:, 0]) - zoom_radius, max(all_points[:, 0]) + zoom_radius
    y_min, y_max = min(all_points[:, 1]) - zoom_radius, max(all_points[:, 1]) + zoom_radius

    for ax in fig.axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        if flip_y:
            ax.invert_yaxis()

    if scalebar is not None:
        add_scalebar(fig.axes[ncols - 1], matchx=False, matchy=False, sizex=scalebar['size'], labelx=scalebar['label'], hidex=False, hidey=False, loc='lower left', fontsize=7, 
                        barwidth=1, offset=scalebar['offset'])

    if show:
        plt.show()
    return fig

def plot_radial_proportions(prop_df, ax=None, legend=True, bbox_to_anchor=(1.1, 1.1), fontsize=6, pad=0.1, colors=None, label_colors=None):
    import itertools

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    categories = prop_df.index.tolist()
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    if len(prop_df.columns) == 1:
        values = prop_df.iloc[:, 0].tolist()
        all_vals = np.array(values)
    else:
        values = prop_df.to_dict(orient='list')
        all_vals = np.concatenate(list(values.values()))

    # ensure finite
    all_vals = all_vals[np.isfinite(all_vals)]

    # dynamic limits (log values)
    rmin = np.floor(np.min(all_vals))
    rmax = np.ceil(np.max(all_vals))
    ax.set_ylim(rmin, rmax)

    # Values (close the polygon)
    if isinstance(values, dict):
        if colors is None:
            colors = itertools.cycle(['mediumblue', 'mediumvioletred', 'darkorange', 'yellowgreen'])
        else:
            colors = itertools.cycle(colors)
        for key, value in values.items():
            value += value[:1]
            color = next(colors)
            ax.plot(angles, value, color=color, linewidth=2, label=key)
            ax.fill(angles, value, color=color, alpha=0.25)
    else:
        values += values[:1]
        ax.plot(angles, values, color='b', linewidth=1)
        ax.fill(angles, values, color='b', alpha=0.25)

    if legend:
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc='upper left')

    # Labels
    r_labels = [0.001, 0.01, 0.1, 1]
    r_ticks = np.log10(r_labels)  # convert to log10 for placement

    ax.set_yticks(r_ticks)        # set tick locations
    ax.set_yticklabels(r_labels, fontsize=fontsize)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=fontsize)
    
    angles = ax.get_xticks()
    labels = ax.get_xticklabels()
    for label, angle in zip(labels, angles):
        angle_deg = np.degrees(angle) % 360

        # rotation so text is always roughly upright
        rotation = angle_deg if angle_deg <= 180 else angle_deg - 180
        label.set_rotation(rotation)
        label.set_rotation_mode("anchor")

        # alignment: right side vs left side of the circle
        if 90 < angle_deg < 270:
            label.set_horizontalalignment('right')
        else:
            label.set_horizontalalignment('left')

        x, y = label.get_position()   # (θ, r) in polar coords
        
        label.set_position((x, y+pad))

        if label_colors is not None:
            dot_radius = ax.get_rmax() 
            label_color = label_colors[label.get_text()]
            ax.plot(angle, dot_radius, 'o', color=label_color, markersize=10, markeredgecolor='k', linewidth=0.5)

    return ax
    
def add_region_colormap(obs, groups, ax, width='100%', height='5%', pad=0):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    group_obs = obs[obs['Group'].isin(groups)]
    regions = ['STR', 'GPe', 'GPi', 'SN', 'STH']
    group_obs = group_obs[group_obs['structure'].isin(regions)]
    region_z = group_obs.groupby(['z_order', 'structure']).size().reset_index(name='count')
    pivot = region_z.pivot(index="z_order", columns="structure", values="count")
    pivot[pivot.isna()] = 0
    pivot[pivot > 0] = 1
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)
    pivot_norm = pivot_norm[regions] # set column order for colors
    cax = inset_axes(ax,
                    width=width,
                    height=height,
                    loc="upper center",
                    borderpad=pad)
    pivot_norm.plot(kind='bar', stacked=True, ax=cax, width=1, edgecolor='black', linewidth=0.5, legend=False)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.axis("off")
