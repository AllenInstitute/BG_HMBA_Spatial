import warnings

import numpy as np
import pandas as pd
from pathlib import Path
import geojson
import json
from matplotlib.path import Path as MplPath

def add_centered_rotated_coords_marmoset(
    marm_adata,
    zcol='brain_section_barcode',
    overwrite=False,
    buffer_frac=0.05,
    inplace=True
):
    ''' 
    Translate the rotated coordinates of each marmoset section so that they're
    all roughly centered over each other in the positive quadrant.
    '''

    # handle inplace vs copy, checking for views
    if not inplace:
        marm_adata = marm_adata.copy()
    elif marm_adata.is_view:
        warnings.warn("marm_adata is a view. Creating & returning a copy.")
        marm_adata = marm_adata.copy()

    if 'x_rotated' not in marm_adata.obs.columns or 'y_rotated' not in marm_adata.obs.columns:
        raise ValueError("marm_adata must have 'x_rotated' and 'y_rotated' columns in .obs")
    elif 'x_rotated_centered_um' in marm_adata.obs.columns and 'y_rotated_centered_um' in marm_adata.obs.columns:
        if not overwrite:
            warnings.warn("marm_adata already has 'x_rotated_centered_um' and 'y_rotated_centered_um' columns in .obs. Skipping this calculation.")
            return marm_adata
        else:
            warnings.warn("Overwriting existing 'x_rotated_centered_um' and 'y_rotated_centered_um' columns in .obs.")

    # Calculate the maximum extent of each section in the rotated coordinates
    sec_extent_x = []
    sec_extent_y = []
    for z in marm_adata.obs[zcol].unique():
        sec_adata = marm_adata[marm_adata.obs[zcol] == z]
        rotated_coords = sec_adata.obs[['x_rotated', 'y_rotated']].values
        sec_extent_x.append(rotated_coords[:,0].max() - rotated_coords[:,0].min())
        sec_extent_y.append(rotated_coords[:,1].max() - rotated_coords[:,1].min())
    
    # Find the maximum extent across all sections, and
    # add a buffer to ensure no cells have negative values after translation
    max_extent_x = max(sec_extent_x)*(1+buffer_frac)
    max_extent_y = max(sec_extent_y)*(1+buffer_frac)

    # Center and translate each section so they all align in the positive quadrant
    for z in marm_adata.obs[zcol].unique():
        sec_adata = marm_adata[marm_adata.obs[zcol] == z]
        rotated_coords = sec_adata.obs[['x_rotated', 'y_rotated']].values

        # center the rotated coordinates on (0,0) using the middle point of each section
        center_of_mass = np.median(rotated_coords, axis=0)
        centered_coords = rotated_coords - center_of_mass

        # translate the centered coordinates so all sections are in the positive quadrant
        translated_coords = centered_coords + np.array([max_extent_x/2, max_extent_y/2])

        marm_adata.obs.loc[sec_adata.obs.index, 'x_rotated_centered_um'] = translated_coords[:,0]
        marm_adata.obs.loc[sec_adata.obs.index, 'y_rotated_centered_um'] = translated_coords[:,1]

    if not inplace:
        return marm_adata
    else:
        return None

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
            rotation_matrix: 3x3 homogeneous transformation matrix used for rotation and translation
    '''   
    # Calculate rotation angle to make the midline vertical
    # - add π/2 to get perpendicular angle for vertical orientation    
    angle = np.arctan2(input_line_yx[1,0]-input_line_yx[0,0],
                       input_line_yx[1,1]-input_line_yx[0,1]) + np.pi/2
    rotation_matrix = np.array([[np.cos(angle),-np.sin(angle)],
                                [np.sin(angle),np.cos(angle)]])

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
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        rotated_line = np.dot(input_line_yx[:,::-1], rotation_matrix)
        rotated_coordinates = np.dot(input_coords,rotation_matrix)  

    # Translate coordinates so midline is positioned at x=0
    tx = -rotated_line[0, 0]
    ty = 0
    rotated_coordinates[:,0] += tx
    rotated_coordinates[:,1] += ty
    
    # Generate column-vector-compatible homogeneous (3x3) affine matrix
    rotation_matrix = np.array([[rotation_matrix[0, 0], rotation_matrix[1,0], tx],
                                [rotation_matrix[0, 1], rotation_matrix[1, 1], ty],
                                [0, 0, 1]])

    return dict(rotated_line=rotated_line,
                rotated_coordinates=rotated_coordinates,
                rotation_matrix=rotation_matrix)

def transform_table(transform_subset: list[dict], table: pd.DataFrame, mask_dir: Path, x_colname: str, y_colname: str, scale_factor: int = 20):
    """Given a transform manifest and a cell-by-gene or binned/gridded AnnData,
    apply an affine transform and save the coordinates in obs."""
    
    table['x_slab'] = np.zeros(table.shape[0])
    table['y_slab'] = np.zeros(table.shape[0])
    table['subset_label'] = np.zeros(table.shape[0])

    for row in transform_subset:
        # load subset mask
        subset_label = row.get('subset_label', 'nan')

        if subset_label != 'nan':
            source = row['source']
            subset_mask = np.load(mask_dir.joinpath(f'{source}_{subset_label}.npy'))
            # convert to bool
            subset_mask = subset_mask != 0
            subset_mask = subset_mask.T

            values_at_coords = mask_coordinates(
                table[x_colname].astype(float),
                table[y_colname].astype(float),
                scale_factor,
                subset_mask
                )

            # values of 1 mean the point is in the mask for this subset
            table.loc[values_at_coords, 'subset_label'] = subset_label
        
        else:
            table['subset_label'] = 'nan'

        data_mask = table['subset_label'] == subset_label
        data_subset = table[data_mask]
        transform = row['transform']
        # transform the data (may have points not included in the reference)
        coords = np.array([[x,y] for x,y in zip(data_subset[x_colname], data_subset[y_colname])])
        transformed_coords = transform_coordinates(coords, transform)

        table.loc[data_mask, 'x_slab'] = transformed_coords[:, 0]
        table.loc[data_mask, 'y_slab'] = transformed_coords[:, 1]
        table.loc[data_mask, 'y_slab_flipped'] = -transformed_coords[:, 1]

    return table

def mask_points_with_polygon(xy_points, polygon_coords):
    """
    Mask a point cloud to only retain points inside a polygon outline.
    Optimized for large datasets with progress tracking.
    
    Parameters:
    - points: numpy array of shape (N, 2) or (N, 3) containing point coordinates
    - polygon_coords: list of [x, y] coordinates defining the polygon boundary
    - show_progress: bool, whether to show progress bar
    
    Returns:
    - masked_points: numpy array containing only points inside the polygon
    - mask: boolean array indicating which points are inside the polygon
    """
    
    # Create a matplotlib Path object for efficient point-in-polygon testing
    polygon_path = MplPath(polygon_coords)
    
    # For very large datasets, process in chunks to manage memory and show progress
    chunk_size = 1000000  # Process 1M points at a time
    total_points = len(xy_points)
    
    if total_points <= chunk_size:
        # Small dataset - process all at once
        mask = polygon_path.contains_points(xy_points)
    else:
        # Large dataset - process in chunks
        mask = np.zeros(total_points, dtype=bool)
        
        for i in range(0, total_points, chunk_size):
            end_idx = min(i + chunk_size, total_points)
            chunk_mask = polygon_path.contains_points(xy_points[i:end_idx])
            mask[i:end_idx] = chunk_mask
    
    # Apply the mask to get only points inside the polygon
    masked_points = xy_points[mask]
    
    return masked_points, mask

def add_aligned_rotated_coords_marmoset(
    marm_adata,
    zcol='brain_section_barcode',
    overwrite=False,
    buffer_frac=0.05,
    inplace=True
):
    ''' 
    Translate the rotated coordinates of each marmoset section so that they're
    all roughly centered over each other in the positive quadrant.
    '''

    # handle inplace vs copy, checking for views
    if not inplace:
        marm_adata = marm_adata.copy()
    elif marm_adata.is_view:
        warnings.warn("marm_adata is a view. Creating & returning a copy.")
        marm_adata = marm_adata.copy()

    if 'x_rotated' not in marm_adata.obs.columns or 'y_rotated' not in marm_adata.obs.columns:
        raise ValueError("marm_adata must have 'x_rotated' and 'y_rotated' columns in .obs")
    elif 'x_rotated_aligned' in marm_adata.obs.columns and 'y_rotated_aligned' in marm_adata.obs.columns:
        if not overwrite:
            warnings.warn("marm_adata already has 'x_rotated_aligned' and 'y_rotated_aligned' columns in .obs. Skipping this calculation.")
            return marm_adata
        else:
            warnings.warn("Overwriting existing 'x_rotated_aligned' and 'y_rotated_aligned' columns in .obs.")

    # Calculate the maximum extent of each section in the rotated coordinates
    sec_extent_x = []
    sec_extent_y = []
    for z in marm_adata.obs[zcol].unique():
        sec_adata = marm_adata[marm_adata.obs[zcol] == z]
        rotated_coords = sec_adata.obs[['x_rotated', 'y_rotated']].values
        sec_extent_x.append(rotated_coords[:,0].max() - rotated_coords[:,0].min())
        sec_extent_y.append(rotated_coords[:,1].max() - rotated_coords[:,1].min())
    
    # Find the maximum extent across all sections, and
    # add a buffer to ensure no cells have negative values after translation
    max_extent_x = max(sec_extent_x)*(1+buffer_frac)
    max_extent_y = max(sec_extent_y)*(1+buffer_frac)

    # Center and translate each section so they all align in the positive quadrant
    for z in marm_adata.obs[zcol].unique():
        sec_adata = marm_adata[marm_adata.obs[zcol] == z]
        rotated_coords = sec_adata.obs[['x_rotated', 'y_rotated']].values

        # center the rotated coordinates on (0,0) using the middle point of each section
        center_of_mass = np.median(rotated_coords, axis=0)
        centered_coords = rotated_coords - center_of_mass

        # translate the centered coordinates so all sections are in the positive quadrant
        translated_coords = centered_coords + np.array([max_extent_x/2, max_extent_y/2])

        marm_adata.obs.loc[sec_adata.obs.index, 'x_rotated_aligned'] = translated_coords[:,0]
        marm_adata.obs.loc[sec_adata.obs.index, 'y_rotated_aligned'] = translated_coords[:,1]

    if not inplace:
        return marm_adata
    else:
        return None

def transform_coordinates(coordinates: np.array, transform_matrix: np.array):
    """
    Transform coordinates using an affine transformation matrix
    """
    # add dummy dimension of a value of 1
    coords = np.hstack((coordinates, np.ones((coordinates.shape[0],1))))

    # apply affine transformation to coordinates
    transformed_coords = transform_matrix @ coords.T

    return transformed_coords.T

def mask_coordinates(x_coords, y_coords, scale_factor, subset_mask):
    # subset coordinates
    coords_scaled = np.array([[x,y] for x,y in zip(x_coords/scale_factor, y_coords/scale_factor)])
    coords_scaled_int = np.floor(coords_scaled).astype(int)

    # add some padding to the edge if the mask is too small
    x_max_data = np.max(coords_scaled_int[:,0])
    y_max_data = np.max(coords_scaled_int[:,1])
    x_max_mask = subset_mask.shape[0]
    y_max_mask = subset_mask.shape[1]

    if x_max_data > x_max_mask:
        x_diff = x_max_data - x_max_mask + 1
        subset_mask = np.pad(subset_mask, ((0, x_diff),(0, 0)), 'edge')

    if y_max_data > y_max_mask:
        y_diff = y_max_data - y_max_mask + 1
        subset_mask = np.pad(subset_mask, ((0, 0),(0, y_diff)), 'edge')

    values_at_coords = subset_mask[tuple(coords_scaled_int.T-1)] # not sure why -1 is needed here but hopefully it works...
    return values_at_coords