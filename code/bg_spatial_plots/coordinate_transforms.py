import warnings

import numpy as np

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