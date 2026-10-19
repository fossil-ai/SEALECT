#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

ROOT = Path('/path/to/sealect_root/experiments')
DATA_ROOT = ROOT / 'dataset_decide'
EXP1_ROOT = ROOT / 'experiment1' / 'exp1_retrieval_quality'
TEST_PAIRS_JSON = DATA_ROOT / 'splits_dino_decorrelated_kbalanced_grounded_sam2' / 'test_scene_object_pairs.json'
OUT_ROOT = ROOT / 'experiment3' / 'pose_plausibility_grounded_sam2'

TOP_KS = [20, 100]
ALPHA = 0.5
LOW_PROB_THRESHOLD = 0.01
MODES = ['yaw', 'joint_pose', 'ybin', 'scale_bin']
EPS = 1e-12

METHOD_DIRS = [
    'random_k20','random_k100','clip_k20','clip_k100','dino_k20','dino_k100',
    'clip_prior_k20','clip_prior_k100','dino_prior_k20','dino_prior_k100',
    'prior_only_k20','prior_only_k100',
    'neural_only_dinofeat_k20','neural_only_dinofeat_k100',
    'sealect_dinofeat_k20','sealect_dinofeat_k100',
    'discofos_ft_k20','discofos_ft_k100',
    'neural_gala_k20','neural_gala_k100',
    'gala_only_k20','gala_only_k100',
    'sweep_neural_only','sweep_context_only','sweep_obs_only',
    'sweep_ln1_lc1_lo0p05','sweep_ln1_lc1_lo0p10','sweep_ln1_lc1_lo0p25',
    'sweep_ln1_lc1_lo0p50','sweep_ln1_lc1_lo1p00',
    'sweep_ln0p75_lc1_lo0p10','sweep_ln1p25_lc1_lo0p10',
    'sweep_ln1_lc0p10_lo0','sweep_ln1_lc0p25_lo0','sweep_ln1_lc0p50_lo0',
    'sweep_ln1_lc1p00_lo0','sweep_ln1_lc2p00_lo0',
    'sweep_ln1_lc0p50_lo0p10','sweep_ln1_lc0p50_lo0p25','sweep_ln1_lc2p00_lo0p10',
]
AUTO_INCLUDE_ALL_SWEEPS = True

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

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def context_key(row):
    return str(row.get('context_key', 'missing_context_key'))

def token_for_mode(row, mode):
    if mode == 'yaw':
        return str(int(row['yaw_bin']))
    if mode == 'pitch':
        return str(int(row['pitch_bin']))
    if mode == 'roll':
        return str(int(row['roll_bin']))
    if mode == 'joint_pose':
        return f"{int(row['pitch_bin'])}_{int(row['yaw_bin'])}_{int(row['roll_bin'])}"
    if mode == 'pose_family':
        if row.get('pose_family') is not None:
            return str(row['pose_family'])
        return f"{row.get('pitch_family','missing_pitch')}__{row.get('yaw_family','missing_yaw')}__{row.get('roll_family','missing_roll')}"
    if mode == 'ybin':
        return str(int(row['y_bin']))
    if mode == 'scale_bin':
        area = float(row.get('area_frac', 0.0))
        edges = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
        for i, edge in enumerate(edges):
            if area <= edge:
                return str(i)
        return str(len(edges))
    raise ValueError(f'Unknown mode: {mode}')

def build_gt_counts(test_pairs, mode):
    counts_by_context = defaultdict(Counter)
    global_support = set()
    for row in test_pairs:
        k = context_key(row)
        tok = token_for_mode(row, mode)
        counts_by_context[k][tok] += 1
        global_support.add(tok)
    return counts_by_context, sorted(global_support)

def smoothed_probability(counter, token, support, alpha):
    n = sum(counter.values())
    v = len(support)
    if v == 0:
        return 0.0
    return (counter.get(token, 0) + alpha) / (n + alpha * v)

def clean_method_name(method_dir, top_k):
    suffix = f'_k{top_k}'
    return method_dir[:-len(suffix)] if method_dir.endswith(suffix) else method_dir

def result_path_for(method_dir, top_k):
    return EXP1_ROOT / method_dir / f'topk_{top_k}' / 'retrieval_results.json'

def sort_key(method_dir):
    clean = clean_method_name(clean_method_name(method_dir, 20), 100)
    if clean in METHOD_ORDER:
        return (METHOD_ORDER.index(clean), clean)
    if clean.startswith('sweep_'):
        return (len(METHOD_ORDER), clean)
    return (len(METHOD_ORDER) + 1, clean)

def discover_methods(top_k):
    names = set()
    for m in METHOD_DIRS:
        if result_path_for(m, top_k).exists():
            names.add(m)
    if AUTO_INCLUDE_ALL_SWEEPS:
        for p in sorted(EXP1_ROOT.glob(f'sweep_*/topk_{top_k}/retrieval_results.json')):
            names.add(p.parent.parent.name)
    return sorted(names, key=sort_key)

def evaluate_method(method_dir, result_path, top_k, gt_counts_by_mode, support_by_mode):
    results = load_json(result_path)
    method = clean_method_name(method_dir, top_k)
    summary_rows, context_rows = [], []

    for mode in MODES:
        all_logps, all_probs, all_violations = [], [], []
        per_context = defaultdict(lambda: {'logps': [], 'probs': [], 'violations': []})
        gt_counts_by_context = gt_counts_by_mode[mode]
        support = support_by_mode[mode]

        for item in results:
            q = item.get('query', {})
            k = context_key(q)
            if k not in gt_counts_by_context:
                continue
            gt_counter = gt_counts_by_context[k]
            for cand in item.get('topk', [])[:top_k]:
                try:
                    tok = token_for_mode(cand, mode)
                except Exception:
                    continue
                p = smoothed_probability(gt_counter, tok, support, ALPHA)
                logp = math.log(p + EPS)
                violation = 1 if p < LOW_PROB_THRESHOLD else 0
                all_logps.append(logp); all_probs.append(p); all_violations.append(violation)
                per_context[k]['logps'].append(logp)
                per_context[k]['probs'].append(p)
                per_context[k]['violations'].append(violation)

        summary_rows.append({
            'method': method, 'method_dir': method_dir, 'mode': mode, 'top_k': top_k,
            'num_scored': len(all_logps),
            'mean_log_likelihood': float(np.mean(all_logps)) if all_logps else float('nan'),
            'median_log_likelihood': float(np.median(all_logps)) if all_logps else float('nan'),
            'mean_probability': float(np.mean(all_probs)) if all_probs else float('nan'),
            'violation_rate': float(np.mean(all_violations)) if all_violations else float('nan'),
        })

        for k, d in per_context.items():
            context_rows.append({
                'method': method, 'method_dir': method_dir, 'mode': mode, 'top_k': top_k,
                'context_key': k, 'num_scored': len(d['logps']),
                'mean_log_likelihood': float(np.mean(d['logps'])) if d['logps'] else float('nan'),
                'median_log_likelihood': float(np.median(d['logps'])) if d['logps'] else float('nan'),
                'mean_probability': float(np.mean(d['probs'])) if d['probs'] else float('nan'),
                'violation_rate': float(np.mean(d['violations'])) if d['violations'] else float('nan'),
            })

    return summary_rows, context_rows

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not TEST_PAIRS_JSON.exists():
        raise FileNotFoundError(f'Missing test pairs: {TEST_PAIRS_JSON}')
    if not EXP1_ROOT.exists():
        raise FileNotFoundError(f'Missing Exp. 1 root: {EXP1_ROOT}')

    test_pairs = load_json(TEST_PAIRS_JSON)
    gt_counts_by_mode, support_by_mode = {}, {}
    for mode in MODES:
        counts, support = build_gt_counts(test_pairs, mode)
        gt_counts_by_mode[mode] = counts
        support_by_mode[mode] = support

    all_summary, all_context, missing = [], [], []
    print('Evaluating Exp. 3 pose plausibility')
    print(f'Exp1 root: {EXP1_ROOT}')
    print(f'Test pairs: {TEST_PAIRS_JSON}')

    for top_k in TOP_KS:
        out_dir = OUT_ROOT / f'k{top_k}'
        out_dir.mkdir(parents=True, exist_ok=True)
        methods = discover_methods(top_k)
        print(f'\nTOP_K={top_k}; methods found: {len(methods)}')
        summary_rows, context_rows = [], []
        for method_dir in methods:
            path = result_path_for(method_dir, top_k)
            if not path.exists():
                missing.append({'method_dir': method_dir, 'top_k': top_k, 'path': str(path)})
                continue
            print(f'  Evaluating {method_dir}: {path}')
            s, c = evaluate_method(method_dir, path, top_k, gt_counts_by_mode, support_by_mode)
            summary_rows.extend(s); context_rows.extend(c)

        save_csv(summary_rows, out_dir / 'plausibility_summary.csv')
        save_csv(context_rows, out_dir / 'plausibility_by_context.csv')
        save_json(summary_rows, out_dir / 'plausibility_summary.json')
        save_json(context_rows, out_dir / 'plausibility_by_context.json')
        all_summary.extend(summary_rows); all_context.extend(context_rows)
        print(f"  Saved: {out_dir / 'plausibility_summary.csv'}")

    save_csv(all_summary, OUT_ROOT / 'plausibility_summary_all.csv')
    save_csv(all_context, OUT_ROOT / 'plausibility_by_context_all.csv')
    save_json(all_summary, OUT_ROOT / 'plausibility_summary_all.json')
    save_json(all_context, OUT_ROOT / 'plausibility_by_context_all.json')
    if missing:
        save_json(missing, OUT_ROOT / 'missing_methods.json')
        print('\nMissing files:')
        for item in missing:
            print(f"  {item['method_dir']} K={item['top_k']}: {item['path']}")
    print('\nDone.')
    print(f"All summary: {OUT_ROOT / 'plausibility_summary_all.csv'}")

if __name__ == '__main__':
    main()
