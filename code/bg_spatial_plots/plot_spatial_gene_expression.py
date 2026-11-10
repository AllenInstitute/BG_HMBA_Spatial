import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def multi_species_dotplot(
    species_adatas,
    genes,
    groupby_key='Group',
    layer='log2cpt',
    group_order=None,
    group_name_map=None,
    group_colors=None,
    species_colors=None,
    max_dot_size=200,
    cmap='Reds',
    figsize=None,
    layout='species_rows',
):
    # ── defaults ──────────────────────────────────────────────────────────────
    if group_order is None:
        seen = {}
        for _, adata in species_adatas:
            for g in adata.obs[groupby_key].unique():
                seen[g] = True
        group_order = sorted(seen.keys())

    if group_name_map is None:
        group_name_map = {}
    if group_colors is None:
        group_colors = {}
    if species_colors is None:
        default_palette = ['#4C9ED9', '#E07B39', '#6ABF69', '#9B59B6', '#E74C3C']
        species_colors = {
            label: default_palette[i % len(default_palette)]
            for i, (label, _) in enumerate(species_adatas)
        }

    species_labels = [label for label, _ in species_adatas]
    n_species = len(species_adatas)
    n_groups  = len(group_order)
    n_genes   = len(genes)

    # ── compute expression stats ──────────────────────────────────────────────
    mean_expr = {}
    frac_expr = {}
    for species_label, adata in species_adatas:
        for grp in group_order:
            mask = adata.obs[groupby_key] == grp
            for gene in genes:
                if mask.sum() == 0 or gene not in adata.var_names:
                    mean_expr[(species_label, grp, gene)] = np.nan
                    frac_expr[(species_label, grp, gene)] = np.nan
                else:
                    if layer is not None:
                        raw  = adata[mask, gene].layers[layer]
                    else:
                        raw  = adata[mask, gene].X
                    vals = np.asarray(
                        raw.todense() if hasattr(raw, 'todense') else raw
                    ).flatten()
                    mean_expr[(species_label, grp, gene)] = vals.mean()
                    frac_expr[(species_label, grp, gene)] = (vals > 0).mean()

    all_means = [v for v in mean_expr.values() if not np.isnan(v)]
    vmax    = max(all_means) if all_means else 1.0
    norm    = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap_fn = plt.get_cmap(cmap)

    def dot_color(sp, grp, gene):
        v = mean_expr.get((sp, grp, gene), np.nan)
        return cmap_fn(norm(v)) if not np.isnan(v) else None

    def dot_size(sp, grp, gene):
        v = frac_expr.get((sp, grp, gene), np.nan)
        return v * max_dot_size if not np.isnan(v) else None

    # ── legend helper ─────────────────────────────────────────────────────────
    def _draw_legends(fig):
        fig.subplots_adjust(bottom=0.22)
        cbar_ax = fig.add_axes([0.08, 0.06, 0.30, 0.03])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f'Mean {layer}', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        size_ax = fig.add_axes([0.50, 0.01, 0.45, 0.12])
        size_ax.set_axis_off()
        for x, frac, label in zip(
            np.linspace(0.05, 0.80, 4),
            [0.25, 0.50, 0.75, 1.0],
            ['25%', '50%', '75%', '100%'],
        ):
            size_ax.scatter(x, 0.65, s=frac * max_dot_size,
                            color='grey', edgecolors='grey', linewidths=0.4,
                            transform=size_ax.transAxes, clip_on=False)
            size_ax.text(x, 0.15, label, ha='center', va='top',
                         fontsize=9, transform=size_ax.transAxes)
        size_ax.text(0.42, 1.05, 'Fraction expressing',
                     ha='center', va='bottom', fontsize=11,
                     transform=size_ax.transAxes)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYOUT: species_rows
    # ══════════════════════════════════════════════════════════════════════════
    if layout == 'species_rows':
        all_rows = []
        for sp_label, adata in species_adatas:
            for grp in group_order:
                if (adata.obs[groupby_key] == grp).sum() > 0:
                    all_rows.append((sp_label, grp))

        n_rows = len(all_rows)
        if figsize is None:
            figsize = (n_genes * 0.9 + 2.5, n_rows * 0.35 + 1.8)
        fig, ax = plt.subplots(figsize=figsize)

        for r, (sp, grp) in enumerate(all_rows):
            for c, gene in enumerate(genes):
                color = dot_color(sp, grp, gene)
                size  = dot_size(sp, grp, gene)
                if color is None:
                    continue
                ax.scatter(c, r, s=size, color=color,
                           edgecolors='grey', linewidths=0.4, zorder=3)

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([group_name_map.get(grp, grp) for _, grp in all_rows], fontsize=10)
        for tick, (sp, grp) in zip(ax.get_yticklabels(), all_rows):
            tick.set_color(group_colors.get(grp, species_colors[sp]))

        prev_sp = None
        for r, (sp, _) in enumerate(all_rows):
            if sp != prev_sp and r > 0:
                ax.axhline(r - 0.5, color='lightgrey', linewidth=0.8, zorder=1)
            prev_sp = sp

        start = 0
        for sp, grp_iter in groupby(all_rows, key=lambda x: x[0]):
            count = sum(1 for _ in grp_iter)
            ax.text(n_genes + 0.1, start + count / 2 - 0.5, sp,
                    va='center', ha='left', fontsize=10,
                    color=species_colors[sp], fontweight='bold')
            start += count

        ax.set_xticks(range(n_genes))
        ax.set_xticklabels(genes, rotation=45, ha='left', fontsize=10)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlim(-0.5, n_genes - 0.5 + 1.8)
        ax.set_ylim(n_rows - 0.5, -0.5)

        for c in range(n_genes):
            ax.axvline(c, color='lightgrey', linewidth=0.5, zorder=0)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYOUT: species_cols  (groups on y, gene × species sub-columns on x)
    # ══════════════════════════════════════════════════════════════════════════
    elif layout == 'species_cols':
        gap = 1

        def x_pos(gene_idx, species_idx):
            return gene_idx * (n_species + gap) + species_idx

        if figsize is None:
            figsize = (x_pos(n_genes - 1, n_species - 1) * 0.4 + 2.0,
                       n_groups * 0.45 + 1.8)
        fig, ax = plt.subplots(figsize=figsize)

        for r, grp in enumerate(group_order):
            for gi, gene in enumerate(genes):
                for si, sp in enumerate(species_labels):
                    color = dot_color(sp, grp, gene)
                    size  = dot_size(sp, grp, gene)
                    if color is None:
                        continue
                    ax.scatter(x_pos(gi, si), r, s=size, color=color,
                               edgecolors='grey', linewidths=0.4, zorder=3)

        ax.set_yticks(range(n_groups))
        ax.set_yticklabels([group_name_map.get(g, g) for g in group_order], fontsize=8)
        for tick, grp in zip(ax.get_yticklabels(), group_order):
            tick.set_color(group_colors.get(grp, 'black'))

        gene_centres = [x_pos(gi, (n_species - 1) / 2) for gi in range(n_genes)]
        ax.set_xticks(gene_centres)
        ax.set_xticklabels(genes, rotation=45, ha='left', fontsize=9)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        # species single-letter sub-labels
        for gi in range(n_genes):
            for si, sp in enumerate(species_labels):
                ax.text(x_pos(gi, si), -0.7, sp[0],
                        ha='center', va='bottom', fontsize=6,
                        color=species_colors[sp], fontweight='bold',
                        transform=ax.get_xaxis_transform())

        # species key
        for si, sp in enumerate(species_labels):
            ax.text(x_pos(0, si), -1.6, f'{sp[0]}={sp}',
                    ha='left', va='bottom', fontsize=7,
                    color=species_colors[sp], fontweight='bold',
                    transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.5, x_pos(n_genes - 1, n_species - 1) + 0.5)
        ax.set_ylim(n_groups - 0.5, -0.5)

        # sub-column grid lines
        for gi in range(n_genes):
            for si in range(n_species):
                ax.axvline(x_pos(gi, si), color='lightgrey', linewidth=0.5, zorder=0)

        # block dividers
        for gi in range(1, n_genes):
            ax.axvline(x_pos(gi, 0) - gap / 2 - 0.5,
                       color='darkgrey', linewidth=0.8, zorder=1)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYOUT: species_cols_T  (genes on y, group × species sub-columns on x)
    # ══════════════════════════════════════════════════════════════════════════
    elif layout == 'species_cols_T':
        gap = 1

        def x_pos(group_idx, species_idx):
            return group_idx * (n_species + gap) + species_idx

        if figsize is None:
            figsize = (x_pos(n_groups - 1, n_species - 1) * 0.4 + 2.0,
                       n_genes * 0.45 + 1.8)
        fig, ax = plt.subplots(figsize=figsize)

        for c, gene in enumerate(genes):
            for gi, grp in enumerate(group_order):
                for si, sp in enumerate(species_labels):
                    color = dot_color(sp, grp, gene)
                    size  = dot_size(sp, grp, gene)
                    if color is None:
                        continue
                    ax.scatter(x_pos(gi, si), c, s=size, color=color,
                               edgecolors='grey', linewidths=0.4, zorder=3)

        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(genes, fontsize=9)

        group_centres = [x_pos(gi, (n_species - 1) / 2) for gi in range(n_groups)]
        ax.set_xticks(group_centres)
        ax.set_xticklabels(
            [group_name_map.get(g, g) for g in group_order],
            ha='center', fontsize=10)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        for tick, grp in zip(ax.get_xticklabels(), group_order):
            tick.set_color(group_colors.get(grp, 'black'))

        # sub-column grid lines
        for gi in range(n_groups):
            for si in range(n_species):
                ax.axvline(x_pos(gi, si), color='lightgrey', linewidth=0.5, zorder=0)

        # block dividers
        for gi in range(1, n_groups):
            ax.axvline(x_pos(gi, 0) - gap / 2 - 0.5,
                       color='darkgrey', linewidth=0.8, zorder=1)

    else:
        raise ValueError(f"Unknown layout {layout!r}. "
                         "Choose 'species_rows', 'species_cols', or 'species_cols_T'.")

    # ── shared axis styling ───────────────────────────────────────────────────
    ax.set_frame_on(False)
    ax.tick_params(length=0)

    _draw_legends(fig)
    return fig, ax