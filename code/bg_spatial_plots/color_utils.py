import colorcet as cc
import seaborn as sns

def generate_color_palette(
    categories, 
    palette_to_match=None, 
    fallback_palette=None,
):
    """
    Fill in missing colors for categories that don't have assigned colors.
    
    Parameters:
    -----------
    categories (list or set): List of all categories that need colors
    palette_to_match (dict): Existing color dictionary to match. Keys are some 
        or all of the `categories`.
    fallback_palettes (str, list of str, or list of hex colors): Colors or 
        colormap to use if color_palette doesn't have enough colors; will re-cycle 
        through the colors as needed. Defaults to colorcet.glasbey, but can be:
        - str: named matplotlib colormap (e.g., 'tab20')
        - list of str: named matplotlib colormaps (e.g., ['tab20b', 'tab20c'])
        - list of hex colors: hex color codes (e.g., ['#ff0000', '#00ff00'])
    
    Returns:
    --------
    dict
        Color dictionary mapping each category to a hex color
    
    Raises:
    -------
    ValueError
        If nested lists are provided or invalid colormap names are given
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    # Handle param formats
    if not isinstance(categories, set):
        categories = set(categories)
    if isinstance(fallback_palette, str):
        fallback_palettes = [fallback_palette]
    else:
        fallback_palettes = fallback_palette

    # Start with the palette_to_match & identify missing categories
    if palette_to_match is not None:
        generated_palette = palette_to_match.copy()
        missing_categories = list(categories.difference(palette_to_match.keys()))
        if missing_categories == []:
            # we can return early if all categories are already present
            return generated_palette
    else:
        generated_palette = {}
        missing_categories = list(categories)

    # Get a "fallback" list of hex color strings from fallback_palettes
    fallback_colors = []
    # if None, use colorcet's glasbey palette
    if fallback_palettes is None:
        glasbey_palette = sns.color_palette(cc.glasbey, n_colors=len(missing_categories))
        fallback_colors = [colors.to_hex(c) for c in glasbey_palette]
    # if a custom list of hex colors, use them directly
    elif all(isinstance(i, str) and i.startswith('#') for i in fallback_palettes):
        fallback_colors = fallback_palettes
    # if all items are valid color strings, convert to hex & use them directly
    elif all(isinstance(i, str) and colors.is_color_like(i) for i in fallback_palettes):
        fallback_colors = [colors.to_hex(i) for i in fallback_palettes]
    # if a list of colormap names, extract colors from each
    elif all(isinstance(i, str) and (i in plt.colormaps()) for i in fallback_palettes):
        for cmap_name in fallback_palettes:
            if cmap_name not in plt.colormaps():
                raise ValueError(f"Invalid matplotlib colormap name: {cmap_name}")
            cmap = plt.get_cmap(cmap_name)
            # if ListedColormap, get all colors directly; otherwise, sample colormap
            if hasattr(cmap, 'colors'):
                fallback_colors.extend([colors.to_hex(c) for c in cmap.colors])
            else:
                n_colors = cmap.N if hasattr(cmap, 'N') and cmap.N < 256 else 256
                fallback_colors.extend([colors.to_hex(cmap(i)) for i in range(n_colors)])
    else:
        raise ValueError(f"Incorrect values for {fallback_palettes}. fallback_palettes must a valid matplotlib color(s)/colormp(s) in the form:\n- a str for a single colormap\n-a single color-like obj\n- a list of str of colormaps\n- a list of color-like objects.")

    # if after all that, we still have no fallback colors, fill with grey
    if not fallback_colors:
        fallback_colors = ['#7F7F7F']  # neutral gray]
    
    # Assign fallback colors to missing categories
    for i, category in enumerate(missing_categories):
        color_idx = i % len(fallback_colors)  # cycle through fallback colors
        generated_palette[category] = fallback_colors[color_idx]
    
    return generated_palette

