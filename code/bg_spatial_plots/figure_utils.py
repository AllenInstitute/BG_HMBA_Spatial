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