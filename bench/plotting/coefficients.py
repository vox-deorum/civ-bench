"""Coefficient / forest-plot helpers (ported from ``shared/plot_utilities.py``).

This is the subset of the old notebook plotting module that the statistics layer
(:mod:`bench.stats`) imports: converting statsmodels coefficient tables into
tidy effect frames and rendering forest plots. The remaining notebook-only
plotting helpers are ported per-analysis as later stages need them.
"""

from __future__ import annotations

import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _show():
    """Display the current figure in notebooks, or call plt.show() interactively."""
    try:
        from IPython.display import display
        display(plt.gcf())
        plt.close()
    except ImportError:
        backend = matplotlib.get_backend()
        if backend and "agg" not in backend.lower():
            plt.show()


def pvalue_to_stars(p):
    """Convert p-value to significance stars: *** (p<0.001), ** (p<0.01), * (p<0.05), or ''."""
    if np.isnan(p):
        return ''
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def clean_variable_name(name, var_type='condition'):
    """
    Clean up variable names from statsmodels coefficient names.

    Handles interaction terms (contains ':') by cleaning each part and
    joining with ' × '. For non-interaction terms, extracts content from
    square brackets [...] if present, otherwise returns the original name.
    """
    # Handle interaction terms: split on ':', clean each part, rejoin
    if ':' in name:
        parts = name.split(':')
        cleaned_parts = [clean_variable_name(p, var_type) for p in parts]
        return ' × '.join(cleaned_parts)

    # Look for pattern [...] and extract what's inside
    match = re.search(r'\[([^\]]+)\]', name)
    if match:
        content = match.group(1)
        if content.startswith('T.') or content.startswith('S.'):
            return content[2:]
        return content

    return name


def log_odds_to_prob_change(log_odds, baseline_prob=0.125):
    """
    Convert log odds ratio to probability change from baseline.

    Returns change in probability as percentage points.
    """
    baseline_odds = baseline_prob / (1 - baseline_prob)
    new_odds = baseline_odds * np.exp(log_odds)
    new_prob = new_odds / (1 + new_odds)
    return (new_prob - baseline_prob) * 100


def deviation_coefficients(model, reference_type):
    """
    Compute deviation-from-grand-mean effects for all player types via contrasts.

    Uses a Treatment-coded OLS model and linear contrasts to get proper CIs/p-values
    for every player type (including the reference). The returned effects are
    deviations from the grand mean, not from the reference category.
    """
    params = model.params
    pt_vars = [c for c in params.index if 'player_type' in c]
    K = len(pt_vars) + 1  # total player types (including reference)

    rows = []
    for var in pt_vars:
        contrast = np.zeros(len(params))
        for j, name in enumerate(params.index):
            if name == var:
                contrast[j] = (K - 1) / K
            elif 'player_type' in name:
                contrast[j] = -1 / K
        t = model.t_test(contrast)
        ci = t.conf_int(alpha=0.05).flatten()
        pval = float(np.squeeze(t.pvalue))
        rows.append({
            'Name': clean_variable_name(var),
            'Effect': float(np.squeeze(t.effect)),
            'CI_Low': float(ci[0]), 'CI_High': float(ci[1]),
            'P_Value': pval,
            'Sig': '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '',
        })

    contrast = np.zeros(len(params))
    for j, name in enumerate(params.index):
        if 'player_type' in name:
            contrast[j] = -1 / K
    t = model.t_test(contrast)
    ci = t.conf_int(alpha=0.05).flatten()
    pval = float(np.squeeze(t.pvalue))
    rows.append({
        'Name': reference_type,
        'Effect': float(np.squeeze(t.effect)),
        'CI_Low': float(ci[0]), 'CI_High': float(ci[1]),
        'P_Value': pval,
        'Sig': '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '',
    })

    df = pd.DataFrame(rows)
    df['Prob_Change'] = np.nan
    df['CI_Low_Prob'] = np.nan
    df['CI_High_Prob'] = np.nan
    return df.sort_values('Effect')


def prepare_coefficient_data(params, conf_int, pvalues, var_names, var_type='condition'):
    """
    Prepare coefficient data for visualization.

    Returns a DataFrame with cleaned names, effects, CIs, significance, and
    probability-change columns.
    """
    data = []
    for var in var_names:
        clean_name = clean_variable_name(var, var_type)
        effect = params[var]
        ci_low, ci_high = conf_int.loc[var]
        pval = pvalues[var]

        if pval < 0.001:
            sig_star = '***'
        elif pval < 0.01:
            sig_star = '**'
        elif pval < 0.05:
            sig_star = '*'
        else:
            sig_star = ''

        data.append({
            'Name': clean_name,
            'Effect': effect,
            'CI_Low': ci_low,
            'CI_High': ci_high,
            'P_Value': pval,
            'Sig': sig_star,
        })

    df = pd.DataFrame(data)
    df['Prob_Change'] = df['Effect'].apply(log_odds_to_prob_change)
    df['CI_Low_Prob'] = df['CI_Low'].apply(log_odds_to_prob_change)
    df['CI_High_Prob'] = df['CI_High'].apply(log_odds_to_prob_change)
    return df.sort_values('Effect')


def plot_forest_plot(df, title, xlabel='Marginal Effect (Relative Rate %)', color='darkblue',
                     figsize=(12, 8), reference_line_label=None,
                     use_prob_scale=True, print_summary=True, sort_alphabetically=False,
                     colors=None, markers=None, annotate_n=False):
    """
    Create a forest plot for regression coefficients with integrated summary.

    Returns matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if use_prob_scale:
        effect_col = 'Prob_Change'
        ci_low_col = 'CI_Low_Prob'
        ci_high_col = 'CI_High_Prob'
        if xlabel == 'Log Odds Ratio':
            xlabel = 'Marginal Effect (Relative Rate %)'
    else:
        effect_col = 'Effect'
        ci_low_col = 'CI_Low'
        ci_high_col = 'CI_High'

    if sort_alphabetically:
        df = df.sort_values('Name')
    else:
        df = df.sort_values(effect_col)

    y_pos = np.arange(len(df))

    for i, row in enumerate(df.itertuples()):
        if colors:
            plot_color = colors.get(row.Name, color)
            alpha = 0.8
        else:
            plot_color = color if row.Sig else 'gray'
            alpha = 0.8 if row.Sig else 0.5
        mk = markers.get(row.Name, 'o') if markers else 'o'

        ax.plot([getattr(row, ci_low_col), getattr(row, ci_high_col)], [i, i],
                color=plot_color, linewidth=2.5, alpha=alpha, solid_capstyle='round')
        ax.scatter(getattr(row, effect_col), i, s=100, color=plot_color,
                   marker=mk, alpha=alpha, zorder=3, edgecolors='black', linewidth=0.5)

        if not colors and row.Sig:
            ax.text(getattr(row, ci_high_col) + (1 if use_prob_scale else 0.05), i, row.Sig,
                    fontsize=12, va='center', color='darkred')

    if annotate_n and 'N' in df.columns:
        x_lo, x_hi = ax.get_xlim()
        x_pad = (x_hi - x_lo) * 0.015
        for i, row in enumerate(df.itertuples()):
            ax.text(getattr(row, ci_high_col) + x_pad, i,
                    f'n={row.N}', fontsize=8, va='center', color='gray')

    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Name'])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if not colors:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor=color, alpha=0.8, label='Significant (p < 0.05)'),
            Patch(facecolor='gray', alpha=0.5, label='Not Significant'),
        ]
        if reference_line_label:
            legend_elements.append(
                Line2D([0], [0], color='red', linestyle='--', alpha=0.5,
                       label=reference_line_label)
            )
        ax.legend(handles=legend_elements, loc='center right')

    plt.tight_layout()

    if print_summary:
        print("\n" + "=" * 60)
        print(f"{title.split(chr(10))[0]} SUMMARY")
        print("=" * 60)

        if colors:
            n_col = 'N' in df.columns
            for _, row in df.iterrows():
                n_str = f'  (n={row["N"]})' if n_col else ''
                print(f"  {row['Name']:30} {row[effect_col]:.4f} [{row[ci_low_col]:.4f}, {row[ci_high_col]:.4f}]{n_str}")
        else:
            if reference_line_label:
                print(f"\nBaseline: {reference_line_label.replace('No effect ', '').replace('Mean ', 'Average of all ')}")

            sig_effects = df[df['Sig'] != ''].copy()
            nonsig_effects = df[df['Sig'] == ''].copy()

            if len(sig_effects) > 0:
                print(f"\nStatistically Significant Effects (p < 0.05):")
                print("-" * 40)
                for _, row in sig_effects.iterrows():
                    if use_prob_scale:
                        print(f"  {row['Name']:30} {row['Prob_Change']:+6.2f}% [{row['CI_Low_Prob']:+6.2f}%, {row['CI_High_Prob']:+6.2f}%] {row['Sig']}")
                    else:
                        print(f"  {row['Name']:30} {row['Effect']:+6.3f} [{row['CI_Low']:+6.3f}, {row['CI_High']:+6.3f}] {row['Sig']}")
            else:
                print("\n  No statistically significant effects found")

            if len(nonsig_effects) > 0:
                print(f"\nNon-Significant Effects:")
                print("-" * 40)
                for _, row in nonsig_effects.iterrows():
                    if use_prob_scale:
                        print(f"  {row['Name']:30} {row['Prob_Change']:+6.2f}% [{row['CI_Low_Prob']:+6.2f}%, {row['CI_High_Prob']:+6.2f}%]")
                    else:
                        print(f"  {row['Name']:30} {row['Effect']:+6.3f} [{row['CI_Low']:+6.3f}, {row['CI_High']:+6.3f}]")

        if not colors:
            print(f"\nOverall Statistics:")
            print("-" * 40)
            print(f"  Total effects analyzed: {len(df)}")
            print(f"  Significant effects: {len(sig_effects)} ({len(sig_effects)/len(df)*100:.1f}%)")
            if use_prob_scale:
                print(f"  Range of effects: {df[effect_col].min():.2f}% to {df[effect_col].max():.2f}%")
                if len(sig_effects) > 0:
                    print(f"  Strongest positive effect: {sig_effects.loc[sig_effects[effect_col].idxmax(), 'Name']} ({sig_effects[effect_col].max():.2f}%)")
                    if sig_effects[effect_col].min() < 0:
                        print(f"  Strongest negative effect: {sig_effects.loc[sig_effects[effect_col].idxmin(), 'Name']} ({sig_effects[effect_col].min():.2f}%)")

    _show()
    return fig, ax
