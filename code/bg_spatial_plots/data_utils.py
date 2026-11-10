from pathlib import Path
import os
import anndata as ad
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import seaborn as sns 
from bg_spatial_plots.coordinate_transforms import add_aligned_rotated_coords_marmoset

NEURON_NEIGHBORHOODS = [
    'Adjacent',
    'Subpallium GABA',
    'Subpallium GABA-Glut',
    'Glut Sero Dopa'
]

def load_anndata_obs(
    ad_filepath,
    data_structure='obs', 
    qc=False, 
    subset=None,
    species=None,
    remove_blanks=False,
):
    '''
    Load just the obs from an AnnData file, with optional quality control and subsetting.
    Optionally load the entire adata object with default filters applied.

    Parameters
    ----------
    ad_file : str
        Path to the AnnData file (.h5ad).
    data_structure : str, {'obs', 'adata'}
        Specify whether to return the 'obs' DataFrame or the full AnnData object.
    qc : bool, default=False
        If True, filter the data to include only cells that pass quality control
        based on 'abc_atlas_visualization' and 'Group_entropy' <= 0.7.
    subset : dict, optional
        A dictionary specifying columns and their corresponding values to subset 
        the data. The keys are column names in `adata.obs`, and the values can 
        be a single value (str, bool, int) or a list of values.
        e.g. subset={'STR': True, 'Neighborhood': 'Nonneuron'}
    species : str, optional, {'human', 'macaque', 'marmoset'}
        Used for species-specific, hard-coded fixes to the data.
    remove_blanks: bool, default=False
        Whether to remove blanks and negative controls from the gene expression 
        matrix.
    
    Returns
    -------
    pd.DataFrame or ad.AnnData
        If `data_structure` is 'obs', returns the filtered `adata.obs` DataFrame.
        If `data_structure` is 'adata', returns the filtered AnnData object.
    '''
    # Load with backing only if we're just returning obs
    if data_structure.lower() == 'obs':
        adata = ad.read_h5ad(ad_filepath, backed='r')
    else:
        adata = ad.read_h5ad(ad_filepath)  # Load into memory
        if remove_blanks:
            blank_mask = (
                adata.var_names.str.contains('(?i)blank', case=False) |   # MERSCOPE blanks
                adata.var_names.str.contains('(?i)codeword', case=False) |  # Xenium blanks
                adata.var_names.str.contains('(?i)control', case=False)  |  # Xenium blanks
                adata.var_names.str.contains('(?i)intergenic', case=False)  # Xenium Prime specific
            )
            adata = adata[:, ~blank_mask].copy()

    # Hardcoded fix for human cluster mis-assignment
    if species and species.lower() == 'human':
        # moving the 'Human-23' cluster from the 'STRd D1 Striosome SPN' group
        # to 'STRd D1 Matrix SPN' group because it's wrong in the reference
        cluster_mask = (adata.obs['Cluster'] == 'Human-23')
        adata.obs.loc[cluster_mask, 'Group'] = 'STRd D1 Matrix SPN'
        if data_structure.lower() == 'adata' and 'DRD2' in adata.var_names:
            # remove DRD2
            warnings.warn(
                "DRD2 is present in the human adata. Removing, but be warned that including DRD2 affects mapping.", 
                stacklevel=2
            ) 
            adata = adata[:, adata.var_names != 'DRD2']
    if species and species.lower() == 'marmoset':
        if 'x_rotated_aligned' not in adata.obs.columns or 'y_rotated_aligned' not in adata.obs.columns:
            print('Aligning rotated coordinates across sections for marmoset...')
            add_aligned_rotated_coords_marmoset(adata, inplace=True)

    # Create combined QC filter + Custom subset mask for filtering the data
    filter_mask = None
    # qc filter
    if qc:
        viz_mask = adata.obs['abc_atlas_visualization']
        # temp fix because marmoset doesn't have this column in there
        if 'Group_entropy' in adata.obs.columns:
            mapping_mask = adata.obs['Group_entropy'] <= 0.7
            qc_mask = viz_mask & mapping_mask
        else:
            qc_mask = viz_mask
        filter_mask = qc_mask
    # custom subset filter
    if subset is not None:
        subset_mask = None
        # loop through the subset dictionary to create a combined mask
        for col, value in subset.items():
            if type(value) == list:
                col_mask = adata.obs[col].isin(value)
            else:
                col_mask = adata.obs[col] == value
            subset_mask = (subset_mask & col_mask) if subset_mask is not None else col_mask
        # add to the existing filter mask
        filter_mask = (filter_mask & subset_mask) if filter_mask is not None else subset_mask

    # Apply filtering
    if filter_mask is not None:
        adata = adata[filter_mask]

    # Decide what to return
    if data_structure.lower() == 'obs':
        data = adata.obs.copy()
    elif data_structure.lower() == 'adata':
        data = adata.copy()
    else:
        raise ValueError("data_structure must be one of {'obs', 'adata'}")

    return data

def load_hemisphere_polygons_marmoset():
    hemi_geojson_dir = Path('/root/capsule/data/marmoset_hmba_bg_spatial_hemisphere_outlines')
    # get list of all geojson files in the directory
    geojson_files = list(hemi_geojson_dir.glob('*.geojson'))

    # Store in a GeoDataFrame
    hemi_polygon_gdf = gpd.GeoDataFrame(
        columns=['brain_section_barcode', 'brain_section_label', 'geometry'],
        crs=None
    )
    for geojson_file in geojson_files:
        sec_gdf = gpd.read_file(geojson_file)
        # add the section labels for easier iterating
        sec_gdf['brain_section_barcode'] = geojson_file.stem.split('_')[-1]  
        sec_gdf['brain_section_label'] = geojson_file.stem.split('_')[-2]
        hemi_polygon_gdf = pd.concat([hemi_polygon_gdf, sec_gdf], ignore_index=True)

    # sort by brain_section_label
    hemi_polygon_gdf = hemi_polygon_gdf.sort_values(by='brain_section_label').reset_index(drop=True)
    return hemi_polygon_gdf

def cell_count_per_structure(obs, anno_cols):
    obs['structure'] = obs[anno_cols].idxmax(axis=1).where(obs[anno_cols].any(axis=1), None)
    obs_anno = obs.groupby(['structure', 'Group']).size().reset_index(name='count')
    obs_anno = obs_anno.pivot(index='Group', columns='structure', values='count')

    return obs_anno

def calculate_group_proportions(obs, group_list, group_column='Group', region=None):
    if isinstance(region, str):
        obs_subset = obs[obs[region]]
    elif isinstance(region, list):
        regions = []
        for r in region:
            regions.append(obs[obs[r]])
        obs_subset = pd.concat(regions)
    else:
        obs_subset = obs.copy()
    obs_subset = obs_subset[obs_subset[group_column].isin(group_list)]
    if obs_subset[group_column].dtype.name == 'category':
        obs_subset[group_column] = obs_subset[group_column].cat.remove_unused_categories()
    if obs_subset[group_column].duplicated().any():
        group_prop = obs_subset.groupby(group_column).size().to_frame('count')
    else: # sometimes the counts are already summed
        group_prop = obs_subset
        group_prop.set_index(group_column, inplace=True)
    group_prop['proportion'] = group_prop['count'] / group_prop['count'].sum()
    group_prop['log_proportion'] = np.log10(group_prop['proportion'])
    
    return group_prop

def group_zorder(obs, groups, windowsize, sort_by='bias', z_score=False, scale=1):
    group_obs = obs[obs['Group'].isin(groups)]
    group_z = group_obs.groupby(['z_order', 'Group']).size().reset_index(name='count')

    pivot = group_z.pivot(index="z_order", columns="Group", values="count")

    smoothed = pd.DataFrame(index=pivot.index.values, columns=groups) 
    stats = pd.DataFrame(index=groups, columns=[sort_by, 'spread'])
    z_orders = group_z['z_order'].unique().tolist()
    z_orders = [z * scale for z in z_orders]
    z_units = z_orders
    if z_score:
        z_mean = np.mean(z_orders)
        z_std = np.std(z_orders)
        z_units = (z_orders - z_mean) / z_std
    for group in groups:
        z_smoothed = pivot[group].rolling(window=windowsize, center=True, min_periods=1).mean()
        z_smoothed_norm = z_smoothed.values / np.sum(z_smoothed.values)
        smoothed[group] = z_smoothed_norm
        if sort_by == 'bias':
            center_of_mass = np.sum(z_units * z_smoothed_norm)
            std = np.sqrt(np.sum(z_smoothed_norm * (z_units - center_of_mass)**2))
            stats.loc[group, sort_by] = center_of_mass
            stats.loc[group, 'spread'] = std
        if sort_by == 'peak':
            peak_pos = np.argmax(z_smoothed_norm)
            stats.loc[group, sort_by] = peak_pos
            stats.loc[group, 'spread'] = None

    sort_order = stats[sort_by].tolist()
    group_order = np.argsort(sort_order)
    smoothed_t = smoothed.T
    smoothed_t = smoothed_t.iloc[group_order]
    return smoothed_t, stats
def bg_rename(
    data,
    anno_lookup_file, 
    levels=None,
):
    """
    Rename taxonomy columns based on annotation lookup table.
    
    Parameters
    ----------
    data : pd.DataFrame or ad.AnnData
        Input DataFrame or AnnData object with taxonomy columns
    anno_lookup_file : str
        Path to annotation lookup CSV file
    levels : list, dict, or None
        Taxonomy levels to rename. If None, auto-detects from data columns.
        - list: ['Group', 'Subclass', ...] assumes col name == taxonomy_level
        - dict: {'Group': 'Group', 'Subclass': 'Subclass'} maps col_name -> taxonomy_level
        - None: auto-detect from standard columns present in data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed taxonomy columns
    """
    # Handle AnnData input
    if hasattr(data, 'obs'):
        obs_copy = data.obs.copy()
    else:
        obs_copy = data.copy()
    
    # Auto-detect taxonomy columns if not specified
    if levels is None:
        standard_cols = ['Neighborhood', 'Class', 'Subclass', 'Group']
        levels = [col for col in standard_cols if col in obs_copy.columns]
        if not levels:
            raise ValueError(
                f"No taxonomy columns found in data. Expected one of {standard_cols}"
            )
    
    # Normalize levels to dict format
    if isinstance(levels, list):
        col_to_level = {col: col for col in levels}
    elif isinstance(levels, dict):
        col_to_level = levels
    else:
        raise TypeError("`levels` must be a list, dict, or None")
    
    # Filter to only columns that exist in data
    col_to_level = {col: lvl for col, lvl in col_to_level.items() if col in obs_copy.columns}

    if not col_to_level:
        raise ValueError(
            f"None of the specified columns exist in data. Found columns: {obs_copy.columns.tolist()}"
        )
    
    # Rename existing columns to temporary names
    obs_copy = obs_copy.rename(columns={c: f'{c}_preprint' for c in col_to_level})
    
    # Process each taxonomy level
    for col_name, taxonomy_level in col_to_level.items():
        original_col = f'{col_name}_preprint'
        if not pd.api.types.is_categorical_dtype(obs_copy[original_col]):
            obs_copy[original_col] = obs_copy[original_col].astype('category')
        original_cats = obs_copy[original_col].cat.categories
        
        # Get rename mapping from utility function
        rename_map = get_taxonomy_mapping(anno_lookup_file, taxonomy_level)
        
        # Validate all categories are in lookup
        missing_cats = [cat for cat in original_cats if cat not in rename_map]
        if missing_cats:
            raise ValueError(
                f"\nMissing taxonomy categories for '{col_name}' (level: {taxonomy_level}):\n"
                f"{missing_cats}\n\nAvailable in lookup: {list(rename_map.keys())}"
            )
        
        # Apply renaming
        new_cats_in_orig_order = [rename_map[cat] for cat in original_cats]
        obs_copy[col_name] = obs_copy[original_col].map(rename_map)
        obs_copy[col_name] = pd.Categorical(
            obs_copy[col_name],
            categories=new_cats_in_orig_order,
            ordered=True
        )
    
    # Clean up temporary columns
    obs_copy = obs_copy.drop(columns=[f'{c}_preprint' for c in col_to_level])
    return obs_copy

def get_taxonomy_mapping(anno_lookup_file, taxonomy_level):
    """
    Create a mapping dictionary from pre-print to final nomenclature for a taxonomy level.
    
    Parameters
    ----------
    anno_lookup_file : str
        Path to annotation lookup CSV file
    taxonomy_level : str
        The taxonomy level to map (e.g., 'Group', 'Subclass', 'Neighborhood')
    
    Returns
    -------
    dict
        Mapping from pre-print_nomenclature -> final_nomenclature
    """
    anno = pd.read_csv(anno_lookup_file)
    
    rename_map = (
        anno[anno['taxonomy_level'] == taxonomy_level]
        .set_index('pre-print_nomenclature')['final_nomenclature']
        .to_dict()
    )
    rename_map['Adjacent'] = 'Adjacent'

    return rename_map