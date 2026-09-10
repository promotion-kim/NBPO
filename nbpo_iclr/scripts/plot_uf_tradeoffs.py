#!/usr/bin/env python3
"""Plot measured four-objective trade-offs; blank inputs produce a draft frame.

Usage: python scripts/plot_uf_tradeoffs.py [--input data/uf_tradeoffs.csv]
       [--output figures/uf_tradeoffs.pdf]

Each row is an aggregate policy point under one common dataset, test hash,
judge and protocol. Dominance uses all FOUR higher-is-better means, not just
the two displayed coordinates. Filled/hollow markers indicate descriptive
dominance of point estimates among the evaluated rows only. They do not imply
statistically significant dominance or population/global Pareto optimality.
"""
from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ('if', 'truth', 'honesty', 'help')


def load(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    measured = []
    for row in rows:
        populated = [bool(row.get(k + '_mean', '').strip()) for k in CRITERIA]
        if any(populated) and not all(populated):
            raise ValueError(f"Incomplete four-objective point: {row['method_id']}")
        if not all(populated):
            continue
        for field in ['dataset', 'test_hash', 'protocol_id', 'judge_id', 'n_prompts', 'n_seeds']:
            if not row.get(field, '').strip():
                raise ValueError(f"Measured point {row['method_id']} requires {field}")
        if int(row['n_prompts']) < 1 or int(row['n_seeds']) < 1:
            raise ValueError('Counts must be positive for measured points.')
        for k in CRITERIA:
            row[k + '_mean'] = float(row[k + '_mean'])
            if not 0 <= row[k + '_mean'] <= 1:
                raise ValueError('All objective axes are win rates in [0, 1].')
            low, high = row.get(k + '_ci_low', '').strip(), row.get(k + '_ci_high', '').strip()
            if bool(low) != bool(high):
                raise ValueError('Provide both CI endpoints or leave both blank.')
            if low:
                low, high = float(low), float(high)
                if not 0 <= low <= row[k + '_mean'] <= high <= 1:
                    raise ValueError('CI must be in [0, 1] and contain the point estimate.')
                if not row.get('uncertainty_type', '').strip():
                    raise ValueError('A plotted CI needs its uncertainty_type.')
                row[k + '_error'] = [[row[k + '_mean'] - low], [high - row[k + '_mean']]]
        measured.append(row)
    for field in ['dataset', 'test_hash', 'protocol_id', 'judge_id']:
        if len({row[field] for row in measured}) > 1:
            raise ValueError(f'Cannot compare rows with different {field}.')
    return measured


def nondominated_mask(values):
    return np.array([not any(np.all(other >= point) and np.any(other > point)
                            for other in values) for point in values])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/uf_tradeoffs.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/uf_tradeoffs.pdf')
    args = parser.parse_args()
    rows = load(args.input)
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3))
    panels = [('help', 'truth', 'Helpfulness win rate', 'Truthfulness win rate'),
              ('if', 'honesty', 'Instruction-following win rate', 'Honesty win rate')]
    mask = nondominated_mask(np.array([[r[k + '_mean'] for k in CRITERIA] for r in rows])) if rows else []
    for ax, (x, y, xlabel, ylabel) in zip(axes, panels):
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.2, linewidth=.4)
        if not rows:
            ax.text(.5, .55, 'DRAFT TEMPLATE', ha='center', va='center', color='.5',
                    fontsize=10, transform=ax.transAxes)
            ax.text(.5, .35, 'Independent four-objective\nevaluations pending', ha='center',
                    va='center', color='.4', fontsize=8, transform=ax.transAxes)
        else:
            for i, (r, is_nondominated) in enumerate(zip(rows, mask)):
                color = plt.get_cmap('tab20')(i % 20)
                ax.errorbar(r[x + '_mean'], r[y + '_mean'],
                            xerr=r.get(x + '_error'), yerr=r.get(y + '_error'),
                            fmt='o' if int(r['n_seeds']) > 1 else '^', ms=4, color=color, capsize=2, elinewidth=.7,
                            mfc=color if is_nondominated else 'white',
                            label=r['method_label'] + f" (s={r['n_seeds']})")
    fig.subplots_adjust(left=.085, right=.98, top=.96, bottom=.27, wspace=.31)
    if rows:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=min(4, len(rows)),
                   frameon=False, fontsize=6.5, bbox_to_anchor=(.5, -.065))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'{args.output}: {len(rows)} measured points; empty input is a draft template.')


if __name__ == '__main__':
    main()
