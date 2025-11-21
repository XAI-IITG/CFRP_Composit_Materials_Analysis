"""Generate synchronized attention vs feature plots (PZT_TOF and Strain_Max).

Saves: `figures/s19_attention_features.png` and prints Pearson r values.

Usage (synthetic):
  python scripts/plot_attention_features.py --synthetic --output figures/s19_attention_features.png

If you have a CSV with columns `cycle`, `attention_weight`, `pzt_tof`, `strain_max`,
call with `--csv path/to.csv` to plot real data.
"""
from __future__ import annotations
import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def synthesize_df(n_cycles: int = 120_000, step: int = 500) -> pd.DataFrame:
    cycles = np.arange(0, n_cycles + 1, step)
    final = cycles.max()

    # Attention proxy: logistic growth centered ~85k
    attention = 1.0 / (1 + np.exp(- (cycles - 85_000) / 8_000))

    # PZT_TOF: smooth rising signal correlated with attention
    pzt_tof = 0.15 + 0.85 * (cycles / final) ** 1.25

    # Strain_Max: rising but with mid-phase bumps and noise
    strain_base = 0.05 + 0.9 * (cycles / final) ** 1.1
    bumps = 0.05 * np.exp(-((cycles - 60_000) ** 2) / (2 * (12_000 ** 2)))
    noise = 0.02 * np.random.RandomState(42).normal(size=cycles.shape)
    strain_max = np.clip(strain_base + bumps + noise, 0, 1)

    df = pd.DataFrame({
        'cycle': cycles,
        'attention_weight': attention,
        'pzt_tof': pzt_tof,
        'strain_max': strain_max,
    })
    return df


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    required = {'cycle'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain column: {required}")
    # Ensure optional columns exist
    for c in ['attention_weight', 'pzt_tof', 'strain_max']:
        if c not in df.columns:
            df[c] = np.nan
    return df[['cycle', 'attention_weight', 'pzt_tof', 'strain_max']]


def pearson_r(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])

def plot_and_save(df: pd.DataFrame, output: str):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    cycles = df['cycle'].values
    attention = df['attention_weight'].values
    
    # --- Color/Style Definitions ---
    ATTENTION_COLOR = '#e76f51'  # Orange/Red
    ATTENTION_STYLE = '--'
    PZT_TOF_COLOR = '#2a9d8f'    # Teal
    STRAIN_MAX_COLOR = '#264653' # Dark Blue
    FEATURE_STYLE = '-'

    # --- Top Plot: PZT_TOF and Attention ---
    ax1 = axes[0]
    # PZT_TOF (Left Y-axis)
    line1 = ax1.plot(cycles, df['pzt_tof'], color=PZT_TOF_COLOR, linestyle=FEATURE_STYLE, label='PZT_TOF')[0]
    ax1.set_ylabel('PZT_TOF (normalized)')
    
    # Attention (Right Y-axis)
    ax1_r = ax1.twinx()
    line1_r = ax1_r.plot(cycles, attention, color=ATTENTION_COLOR, linestyle=ATTENTION_STYLE, label='Attention')[0]
    ax1_r.set_ylabel('Attention (proxy)', color=ATTENTION_COLOR)
    
    # Combined Legend for Top Plot
    lines1 = [line1, line1_r]
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc='upper left')

    r1 = pearson_r(df['pzt_tof'].values, attention)
    ax1.text(0.98, 0.90, f'Pearson r = {r1:.2f}', transform=ax1.transAxes,
              ha='right', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    # --- Bottom Plot: Strain_Max and Attention ---
    ax2 = axes[1]
    # Strain_Max (Left Y-axis)
    line2 = ax2.plot(cycles, df['strain_max'], color=STRAIN_MAX_COLOR, linestyle=FEATURE_STYLE, label='Strain_Max')[0]
    ax2.set_ylabel('Strain_Max (normalized)')
    
    # Attention (Right Y-axis)
    ax2_r = ax2.twinx()
    # Note: No 'label' needed for the second attention line, as the legend is combined.
    line2_r = ax2_r.plot(cycles, attention, color=ATTENTION_COLOR, linestyle=ATTENTION_STYLE)[0]
    ax2_r.set_ylabel('Attention (proxy)', color=ATTENTION_COLOR)
    
    # Combined Legend for Bottom Plot
    # Since the Attention line style is the same, we reuse the first attention line object for the legend.
    lines2 = [line2, line1_r] 
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='upper left')
    
    r2 = pearson_r(df['strain_max'].values, attention)
    ax2.text(0.98, 0.90, f'Pearson r = {r2:.2f}', transform=ax2.transAxes,
              ha='right', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    axes[-1].set_xlabel('Cycles')
    fig.suptitle('Attention vs Feature Dynamics (Specimen S19)')

    # Tight layout and save
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output}")
    print(f"PZT_TOF vs Attention Pearson r: {r1:.3f}")
    print(f"Strain_Max vs Attention Pearson r: {r2:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Optional CSV with cycle, attention_weight, pzt_tof, strain_max')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--output', type=str, default='figures/s19_attention_features.png')
    args = parser.parse_args()

    if args.csv:
        df = load_csv(args.csv)
    elif args.synthetic:
        df = synthesize_df()
    else:
        # Default to synthetic if no input provided
        df = synthesize_df()

    plot_and_save(df, args.output)


if __name__ == '__main__':
    main()
