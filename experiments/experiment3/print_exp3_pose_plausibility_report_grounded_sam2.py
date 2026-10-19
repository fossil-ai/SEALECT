#!/usr/bin/env python3
import csv, math
from pathlib import Path
from collections import defaultdict

ROOT = Path('/path/to/sealect_root/experiments/experiment3/pose_plausibility_grounded_sam2')
TOP_KS = [20, 100]
MODES = ['yaw', 'joint_pose', 'ybin', 'scale_bin']

METHOD_ORDER = [
    'random','clip','dino','clip_prior','dino_prior','prior_only',
    'neural_only_dinofeat','sealect_dinofeat',
    'sweep_neural_only','sweep_context_only','sweep_obs_only',
    'sweep_ln1_lc1_lo0p05','sweep_ln1_lc1_lo0p10','sweep_ln1_lc1_lo0p25',
    'sweep_ln1_lc1_lo0p50','sweep_ln1_lc1_lo1p00',
    'sweep_ln0p75_lc1_lo0p10','sweep_ln1p25_lc1_lo0p10',
    'sweep_ln1_lc0p10_lo0','sweep_ln1_lc0p25_lo0','sweep_ln1_lc0p50_lo0',
    'sweep_ln1_lc1p00_lo0','sweep_ln1_lc2p00_lo0',
    'sweep_ln1_lc0p50_lo0p10','sweep_ln1_lc0p50_lo0p25','sweep_ln1_lc2p00_lo0p10',
]

def read_csv(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f))

def fnum(x, digits=4):
    try:
        x = float(x)
        if math.isnan(x): return 'nan'
        return f'{x:.{digits}f}'
    except Exception:
        return 'nan'

def method_sort_key(m):
    if m in METHOD_ORDER:
        return METHOD_ORDER.index(m)
    if m.startswith('sweep_'):
        return len(METHOD_ORDER)
    return len(METHOD_ORDER) + 1

def main():
    print('\n')
    print('EXPERIMENT 3 PLAUSIBILITY REPORT')
    print('')
    for top_k in TOP_KS:
        csv_path = ROOT / f'k{top_k}' / 'plausibility_summary.csv'
        if not csv_path.exists():
            print(f'\nMissing: {csv_path}')
            continue
        rows = read_csv(csv_path)
        by_mode = defaultdict(list)
        for r in rows:
            by_mode[r['mode']].append(r)
        print(f'\n\n######## K={top_k} ########')
        for mode in MODES:
            subset = by_mode.get(mode, [])
            if not subset: continue
            subset = sorted(subset, key=lambda r: method_sort_key(r['method']))
            print(f'\n--- MODE: {mode} ---')
            print('method'.ljust(30), 'mean_logp'.rjust(12), 'mean_prob'.rjust(12), 'viol_rate'.rjust(12), 'n'.rjust(10))
            print('-' * 84)
            for r in subset:
                print(
                    r['method'].ljust(30),
                    fnum(r['mean_log_likelihood']).rjust(12),
                    fnum(r['mean_probability']).rjust(12),
                    fnum(r['violation_rate']).rjust(12),
                    str(r['num_scored']).rjust(10),
                )
    print('\nLegend:')
    print('  mean_logp  = average log-probability under GT scene-conditioned distribution')
    print('  mean_prob  = average smoothed GT probability of retrieved candidates')
    print('  viol_rate  = fraction of retrieved candidates in very low-probability bins')

if __name__ == '__main__':
    main()
