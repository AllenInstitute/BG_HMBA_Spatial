import json
from pathlib import Path
import os

import geojson
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from spatialdata_io import merscope, xenium, xenium_aligned_image
import spatialdata as sd
from spatialdata.transformations import (
    Affine,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    get_transformation,
    set_transformation,
    remove_transformations_to_coordinate_system
)

def apply_transform_to_merscope_sdata(sdata, section_id, specimen_name, donor, slab, new_cs='slab', rescale=20):
    # Load transform from /data/for_mike/
    fn = donor + '.' + slab
    transform_path = Path(f'/data/for_mike/{fn}.transform_manifest.json')
    with open(transform_path, 'r') as fp:
        transform = json.load(fp)

    for t in transform:
        if t['source'] == specimen_name:
            transform = t
            break

    affine = Affine(transform['transform'], input_axes=('x', 'y'), output_axes=('x', 'y'))
    compensation_factor = Scale([1/9.25936794, 1/9.25936794], ['x', 'y'])
    scale = Scale([rescale, rescale], ['x', 'y'])
    T = Sequence([compensation_factor, affine, scale])

    def transform_element(key):
        if key.startswith(section_id):
            og_T = get_transformation(sdata[key])
            composed = Sequence([og_T, T])
            set_transformation(sdata[key], composed, to_coordinate_system=new_cs)

    for key, element in sdata.labels.items():
        transform_element(key)

    for key, element in sdata.images.items():
        transform_element(key)

    for key, element in sdata.shapes.items():
        transform_element(key)

    for key, element in sdata.points.items():
        transform_element(key)

# this function is different from the MERSCOPE equivalent due to the format of the transform being slightly different
def apply_transform_to_xenium_sdata(sdata, rotation_matrix, new_cs='slab', rescale=1000, flip_axis=None):
    """
    Apply transformation to SpatialData object.
    
    Parameters
    ----------
    flip_axis : str, optional
        Axis to flip ('x', 'y', or None). Applied before rotation.
    """
    affine = Affine(rotation_matrix, input_axes=('x', 'y'), output_axes=('x', 'y'))
    compensation_factor = Scale([1/4.70588235, 1/4.70588235], ['x', 'y'])
    scale = Scale([rescale, rescale], ['x', 'y'])
    
    # Add flip if specified
    transforms = [compensation_factor]
    if flip_axis == 'x':
        transforms.append(Scale([-1, 1], ['x', 'y']))
    elif flip_axis == 'y':
        transforms.append(Scale([1, -1], ['x', 'y']))
    transforms.extend([affine, scale])
    
    T = Sequence(transforms)

    def transform_element(key):
        og_T = get_transformation(sdata[key])
        composed = Sequence([og_T, T])
        set_transformation(sdata[key], composed, to_coordinate_system=new_cs)

    for key, element in sdata.labels.items():
        transform_element(key)

    for key, element in sdata.images.items():
        transform_element(key)

    for key, element in sdata.shapes.items():
        transform_element(key)

    for key, element in sdata.points.items():
        transform_element(key)

def render_shapes_outline_by_column(sdata, ax, shapes_element, table_name, color_column, palette, 
                                     fill_alpha=0.05, outline_width=2):
    """
    Render shape outlines colored by a categorical column in the annotating table.
    
    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to render
    ax : matplotlib.axes.Axes
        The axes to plot on
    shapes_element : str
        Name of the shapes element to render
    table_name : str
        Name of the table with annotations
    color_column : str
        Column in table.obs to use for coloring
    palette : dict
        Dictionary mapping category values to colors
    fill_alpha : float, optional
        Alpha value for shape fills (default: 0.05)
    outline_width : float, optional
        Width of shape outlines (default: 2)
    """
    import spatialdata as sd
    
    table = sdata[table_name]
    shapes = sdata[shapes_element]
    
    for category, color in palette.items():
        # Get cells for this category
        cells_in_category = table.obs[table.obs[color_column] == category].index.tolist()
        
        if len(cells_in_category) > 0:
            # Filter both table and shapes to this category
            temp_table = table[cells_in_category].copy()
            temp_shapes = shapes.loc[cells_in_category]
            
            temp_sdata = sd.SpatialData(
                shapes={shapes_element: temp_shapes},
                tables={table_name: temp_table}
            )
            
            # Render this subset
            temp_sdata.pl.render_shapes(
                element=shapes_element,
                fill_alpha=fill_alpha,
                outline_width=outline_width,
                outline_color=color
            ).pl.show(ax=ax)

def cache_and_load_merscope_zarr(barcode_to_test, merscope_dir):
    # Put the zarr in scratch for now
    # Ok if we don't do this for too many sections
    zarr_path = Path(f'/scratch/{barcode_to_test}_merscope.zarr')
    if not zarr_path.exists():
        # resave as zarr to get that sweet performance
        # this might require a higher memory instance the first time
        sdata = merscope(merscope_dir)
        sdata.write(zarr_path)

        # reload as zarr to improve memory/performance the first time
        del sdata
    
    sdata = sd.read_zarr(zarr_path)
    return sdata

def cache_and_load_xenium_zarr(barcode_to_test, xenium_dir):
    # Put the zarr in scratch for now
    # Ok if we don't do this for too many sections
    zarr_path = Path(f'/scratch/{barcode_to_test}_xenium.zarr')
    if not zarr_path.exists():
        # resave as zarr to get that sweet performance
        # this might require a higher memory instance the first time
        sdata = xenium(xenium_dir, cells_as_circles=False)
        sdata.write(zarr_path)

        # reload as zarr to improve memory/performance the first time
        del sdata
    
    sdata = sd.read_zarr(zarr_path)
    return sdata

def add_fonts_to_manager():
    # locate current conda environment's font folder
    conda_env_path = os.environ.get("CONDA_PREFIX")
    if conda_env_path:
        conda_font_dir = os.path.join(conda_env_path, "fonts")
        
        # find and register every TTF file in that directory
        if os.path.exists(conda_font_dir):
            for font in font_manager.findSystemFonts(fontpaths=[conda_font_dir], fontext='ttf'):
                font_manager.fontManager.addfont(font)

# reproduce midline rotation
# I might've copied from the bg spatial plots repo
def rotate_marmoset_section(
    input_coords, 
    input_line_yx,
):
    ''' Rotate a set of coordinates based on a midline defined by two points.
    **Assumes a right hemisphere section with midline to the left.

    Context: Marmoset sections were manually annotated in napari in the original
    Xenium experiment coordinate space, so anatomical annotation polygons are in
    the same coordinate space as the .obs['x', 'y'] coordinates. To generate the
    .obs['x_rotated', 'y_rotated'] coordinates, a midline was drawn in napari
    and used to orient each section so that the midline is vertical and to the
    left of the section. This function replicates that rotation.
    NB: Two sections, 42.15.06 & 42.16.06, ended up upside-down on the Xenium 
        slides and so are flipped in y after initial rotation.

    Parameters:
        input_coords (np.array, shape (N, 2)): x,y coordinates to be rotated
        input_line_yx: (np.array, shape (2, 2)): y,x coordinates of two points defining the midline
    Returns:
        dict with keys:
            rotated_line: numpy array of shape (2, 2) with rotated midline coordinates
            rotated_coordinates: numpy array of shape (N, 2) with rotated x,y coordinates
    '''   
    # Calculate rotation angle to make the midline vertical
    # - add π/2 to get perpendicular angle for vertical orientation    
    angle = np.arctan2(input_line_yx[1,0]-input_line_yx[0,0],input_line_yx[1,1]-input_line_yx[0,1])+np.pi/2
    rotation_matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

    # Apply initial rotation to both line and coordinates
    # NB: input_line is in y,x order (napari convention), so we reverse the 
    #     order ([:,::-1]) for the proper dot product in x,y order
    rotated_line = np.dot(input_line_yx[:,::-1], rotation_matrix)
    rotated_coordinates = np.dot(input_coords, rotation_matrix)  

    # Check if the cells are to the right of the midline (correct orientation
    # for a right hemisphere section); if not, rotate by 180 degrees (π radians)
    # - Why? B/c two sections, 42.15.06 & 42.16.06, ended up upside-down on the
    #        Xenium slide & need an extra flip in y to be oriented correctly.
    if np.mean(rotated_line[:,0]) > np.mean(rotated_coordinates[:,0]):
        angle = angle-np.pi
        rotation_matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

    return rotation_matrix


def get_annotation_coords(section_id, transform_path, annotation=None, ax=None, swap_coords=False, flip_y=False, scale_factor=1, 
                            plot_args={}, rotate_midline=False):
    transform_path = Path(transform_path)
    if len(list(transform_path.glob('*'))) == 1:
        try:
            transform_file = next(transform_path.glob('*.json'))
            with open(transform_file, 'r') as f:
                annotations = json.load(f)
            if section_id in annotations.keys():
                coords = np.array(annotations[section_id])
            else:
                print (f'no coordinates for section {section_id}')
                return None
                
        except StopIteration:
            print(f'no json file in {transform_path}')
            return None 

    else:
        try:
            transform_file = next(transform_path.glob(f'*{section_id}*'))
            with open(transform_file, 'r') as f:
                annotations = geojson.load(f)
            if annotation is not None:
                anno = [f for f in annotations["features"] if f["properties"]["layer_name"]==annotation]
                if len(anno) == 1:
                    feature = anno[0]
                    try:
                        coords = [np.array(feature["geometry"]["coordinates"])]  
                    except KeyError: # sometimes there are 2 non-contiguous annotations
                        coords = [np.array(g['coordinates'][0]) for g in feature["geometry"]["geometries"]]
                else:
                    print(f'no {annotation} feature for section {section_id}')
                    return None
            else:
                try:
                    # get the first polygon
                    coords = [np.array(annotations['geometries'][0]['coordinates'][0])]
                except KeyError as e:
                    print(f'could not find annotation coordinates \n KeyError: {e}')
                    return None
        except StopIteration:
            print(f'no transform for section {section_id}')
            return None

    coords2 = []
    for c in coords:
        c *= scale_factor

        if c.ndim > 2:
            c = c[0]

        if swap_coords:
            c[:, [0, 1]] = c[:, [1, 0]]

        if flip_y:
            c[:, 1] = -c[:, 1]

        if rotate_midline:
            line_coords = get_annotation_coords(section_id, transform_path, annotation='midline')[0]
            c = rotate_marmoset_section(c, line_coords)['rotated_coordinates']
            
        if ax is not None:
            color = plot_args.get('color', 'grey')
            fill = plot_args.get('fill', False)
            linewidth = plot_args.get('linewidth', 1)
            ax.fill(c[:, 0], c[:, 1], color=color, fill=fill, linewidth=linewidth)
        coords2.append(c)
    
    return coords2