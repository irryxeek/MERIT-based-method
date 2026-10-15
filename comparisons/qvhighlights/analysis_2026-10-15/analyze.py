"""Offline comparison of frozen QVHighlights predictions; never performs inference."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cluster_comparison(left, right, truth, columns):
    qids = sorted(set(left) & set(right))
    groups = sorted({truth[q]['vid'].rsplit('_', 2)[0] for q in qids})
    group_index = {g: i for i, g in enumerate(groups)}
    sums = np.zeros((len(groups), len(columns)))
    counts = np.zeros(len(groups))
    delta = np.array([[left[q][c] - right[q][c] for c in columns] for q in qids])
    for q, values in zip(qids, delta):
        i = group_index[truth[q]['vid'].rsplit('_', 2)[0]]
        sums[i] += values
        counts[i] += 1
    rng = np.random.default_rng(42)
    replicates = []
    for _ in range(100):
        chosen = rng.integers(0, len(groups), size=(100, len(groups)))
        replicates.append(sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)[:, None])
    interval = np.percentile(np.concatenate(replicates), [2.5, 97.5], axis=0)
    return {
        'queries': len(qids), 'source_videos': len(groups),
        'metrics': {c: {
            'delta_pp': float(delta[:, i].mean()),
            'ci95_pp': interval[:, i].tolist(),
            'wins': int((delta[:, i] > 1e-10).sum()),
            'ties': int((np.abs(delta[:, i]) <= 1e-10).sum()),
            'losses': int((delta[:, i] < -1e-10).sum()),
        } for i, c in enumerate(columns)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merit-run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    archive, merit, output = args.archive.resolve(), args.merit_run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(archive / 'common/evaluator'))
    from standalone_eval.eval import compute_ap_from_tuple, compute_hl_ap, compute_hl_hit1, mk_gt_scores

    paths = {'MERIT': merit / 'val_predictions.jsonl'}
    metrics_paths = {'MERIT': merit / 'val_report.json'}
    for protocol, folder in [('compact_v1', 'full_validation_v1'), ('groundvts_v2b', 'full_v2b')]:
        for method in ['global', 'tool']:
            name = f'LongVT_{protocol}_{method}'
            base = archive / f'longvt/{protocol}/results/{folder}/longvt_{method}'
            paths[name] = base / 'predictions.jsonl'
            metrics_paths[name] = base / 'metrics.json'
    paths['VideoARM_HD_partial'] = archive / 'videoarm/runs/formal_c0c8ae9799ad_hd_val_M1/predictions.jsonl'
    gt_path = archive / 'common/ground_truth_val.jsonl'
    truth_list = read_jsonl(gt_path)
    truth = {r['qid']: r for r in truth_list}
    assert len(truth) == len(truth_list) == 1550
    assert truth == {r['qid']: r for r in read_jsonl(merit / 'ground_truth/val.jsonl')}
    dependencies = list(paths.values()) + list(metrics_paths.values()) + [gt_path, merit / 'ground_truth/val.jsonl']
    protocol_path = merit / 'protocol.json'
    evaluator = Path(json.loads(protocol_path.read_text())['config']['evaluator_root'])
    for name in ['eval.py', 'utils.py']:
        saved = archive / 'common/evaluator/standalone_eval' / name
        assert sha256(saved) == sha256(evaluator / 'standalone_eval' / name)
        dependencies.append(saved)
    dependencies.extend([protocol_path, archive / 'alignment/query_alignment.csv'])
    initial_hashes = {str(p): sha256(p) for p in dependencies}
    thresholds = [('Fair', 2), ('Good', 3), ('VeryGood', 4)]
    columns = [f'{name}_{metric}' for name, _ in thresholds for metric in ['mAP', 'Hit1']]
    binaries = {name: {q: (mk_gt_scores(row) >= threshold).astype(float) for q, row in truth.items()}
                for name, threshold in thresholds}
    per_query, summary, prediction_sets = {}, {}, {}
    for name, path in paths.items():
        rows = read_jsonl(path)
        predictions = {r['qid']: r for r in rows}
        assert len(rows) == len(predictions)
        assert set(predictions) <= set(truth)
        assert len(rows) == (419 if name == 'VideoARM_HD_partial' else 1550)
        result = {}
        for q, row in predictions.items():
            assert row['vid'] == truth[q]['vid']
            scores = np.asarray(row['pred_saliency_scores'], dtype=float)
            assert len(scores) == int(truth[q]['duration'] / 2)
            assert np.isfinite(scores).all() and (scores >= 0).all()
            assert (scores <= (100 if name == 'MERIT' else 4)).all()
            top = np.flatnonzero(scores == scores.max())
            item = {'constant': len(set(scores)) == 1, 'top_tie_count': len(top),
                    'unique_scores': len(set(scores))}
            for label, _ in thresholds:
                binary = binaries[label][q]
                ap = [compute_ap_from_tuple((0, w, binary[:, w], scores))[2] for w in range(3)]
                item[f'{label}_mAP'] = float(100 * np.mean(ap))
                item[f'{label}_Hit1'] = float(100 * binary[int(np.argmax(scores))].max())
                item[f'{label}_random_tie_Hit1'] = float(100 * binary[top].max(axis=1).mean())
                item[f'{label}_any_top_hit'] = float(100 * binary[top].max())
            result[q] = item
        official = {}
        for label, _ in thresholds:
            gt_subset = {q: binaries[label][q] for q in predictions}
            official[f'{label}_mAP'] = compute_hl_ap(predictions, gt_subset, num_workers=1)
            official[f'{label}_Hit1'] = compute_hl_hit1(predictions, gt_subset)
        for column in columns:
            assert abs(round(np.mean([r[column] for r in result.values()]), 2) - official[column]) < 1e-8
        if name in metrics_paths:
            stored = json.loads(metrics_paths[name].read_text())
            if name == 'MERIT':
                stored = stored['official_metrics']
            for label, _ in thresholds:
                assert stored[f'HL-min-{label}']['HL-mAP'] == official[f'{label}_mAP']
                assert stored[f'HL-min-{label}']['HL-Hit1'] == official[f'{label}_Hit1']
        summary[name] = {'queries': len(result), 'metrics': official,
                         'constant_queries': sum(r['constant'] for r in result.values()),
                         'tied_maximum_queries': sum(r['top_tie_count'] > 1 for r in result.values()),
                         'mean_top_tie_count': float(np.mean([r['top_tie_count'] for r in result.values()])),
                         'mean_unique_scores': float(np.mean([r['unique_scores'] for r in result.values()])),
                         'diagnostic_random_tie_VeryGood_Hit1': float(np.mean([r['VeryGood_random_tie_Hit1'] for r in result.values()]))}
        per_query[name], prediction_sets[name] = result, predictions
        print(name, json.dumps(summary[name]), flush=True)

    full_pairs = {f'MERIT_minus_{name}': cluster_comparison(per_query['MERIT'], values, truth, columns)
                  for name, values in per_query.items() if name.startswith('LongVT')}
    shared = set(per_query['VideoARM_HD_partial'])
    subset_metrics = {name: {c: float(np.mean([values[q][c] for q in sorted(shared)])) for c in columns}
                      for name, values in per_query.items()}
    pair_partial = cluster_comparison({q: per_query['MERIT'][q] for q in shared},
                                     per_query['VideoARM_HD_partial'], truth, columns)
    records = {q: json.loads((merit / f'queries/{q}.json').read_text()) for q in truth}
    query_hashes = {str(q): sha256(merit / f'queries/{q}.json') for q in sorted(truth)}
    (output / 'merit_query_sha256.json').write_text(json.dumps(query_hashes, indent=2) + '\n')
    pair_native = {q for q in shared if records[q]['status'] == 'ok'}
    pair_native_metrics = {name: {c: float(np.mean([per_query[name][q][c] for q in pair_native]))
                                  for c in columns}
                           for name in ['MERIT', 'VideoARM_HD_partial']}
    with (archive / 'alignment/query_alignment.csv').open() as f:
        aligned = {int(r['qid']): r for r in csv.DictReader(f)}
    native_full = {q for q in truth if records[q]['status'] == 'ok' and all(
        aligned[q][f'longvt_{p}_{m}_fallback'] == 'False'
        for p in ['compact_v1', 'groundvts_v2b'] for m in ['global', 'tool'])}
    native_partial = native_full & shared
    native_subsets = {}
    for label, qids in [('all_five_full_runs_native_success', native_full), ('all_six_native_success', native_partial)]:
        names = [name for name in per_query if label == 'all_six_native_success' or name != 'VideoARM_HD_partial']
        native_subsets[label] = {'queries': len(qids), 'qids': sorted(qids),
                                'metrics': {name: {c: float(np.mean([per_query[name][q][c] for q in qids]))
                                                   for c in columns} for name in names}}
    report = json.loads((merit / 'val_report.json').read_text())
    stage_costs = report['model_requests']
    totals = {key: sum(v[key] for v in stage_costs.values())
              for key in ['logical_requests', 'attempts', 'seconds', 'prompt_tokens', 'completion_tokens',
                          'total_tokens', 'submitted_images_including_retries']}
    merit_diagnostics = {
        'round_histogram': dict(Counter(len(v['rounds']) for v in records.values())),
        'selected_frames_mean_all_queries': float(np.mean([r.get('unique_selected_frames', 0) for r in records.values()])),
        'most_common_scores': Counter(s for r in prediction_sets['MERIT'].values() for s in r['pred_saliency_scores']).most_common(12),
        'cost_totals': totals,
        'memory_request_seconds_fraction': (stage_costs['caption']['seconds'] + stage_costs['keys']['seconds']) / totals['seconds'],
        'memory_total_tokens_fraction': (stage_costs['caption']['total_tokens'] + stage_costs['keys']['total_tokens']) / totals['total_tokens'],
        'score_total_tokens_fraction': stage_costs['score']['total_tokens'] / totals['total_tokens'],
        'reported_mean_query_seconds': report['mean_query_seconds'],
        'timing_note': 'Successful-query timer excludes memory/index; memory failures record time until failure. This mixed mean is not end-to-end latency.',
        'failed_qids': report['failed_qids'],
    }
    # Failure records and query diagnostics are immutable inputs, not inference triggers.
    assert initial_hashes == {str(p): sha256(p) for p in dependencies}
    result = {
        'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope': 'Post-hoc offline comparison; no inference, retraining, or prediction modifications',
        'bootstrap': {'replicates': 10000, 'seed': 42, 'cluster': 'source video: vid.rsplit("_", 2)[0]',
                      'aggregation': 'query-weighted paired mean difference', 'ci': 'percentile 95%',
                      'multiple_comparison_adjustment': False, 'interpretation': 'exploratory; not a method-isolating ablation'},
        'full_validation': {name: v for name, v in summary.items() if name != 'VideoARM_HD_partial'},
        'full_validation_pairs': full_pairs,
        'videoarm_success_selected_subset': {'queries': len(shared), 'qids': sorted(shared),
            'metrics': subset_metrics, 'MERIT_minus_VideoARM': pair_partial,
            'warning': 'Selected by VideoARM completion/success; not the full validation population'},
        'merit_videoarm_pair_native_success': {'queries': len(pair_native), 'qids': sorted(pair_native),
            'metrics': pair_native_metrics,
            'MERIT_minus_VideoARM': cluster_comparison({q: per_query['MERIT'][q] for q in pair_native},
                {q: per_query['VideoARM_HD_partial'][q] for q in pair_native}, truth, columns)},
        'native_success_diagnostics': native_subsets,
        'merit_diagnostics': merit_diagnostics,
        'input_sha256': initial_hashes,
        'script_sha256': sha256(Path(__file__)),
        'merit_query_sha256_file': 'merit_query_sha256.json',
        'verification': 'All five full-run metrics match frozen reports; 419 VideoARM scores newly computed offline; ground truth parsed equality and evaluator byte equality verified; input hashes unchanged',
    }
    (output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (output / 'per_query_metrics.csv').open('w', newline='') as f:
        keys = ['method', 'qid', 'vid', 'source_video'] + list(next(iter(per_query['MERIT'].values())))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for name, values in per_query.items():
            for q, row in sorted(values.items()):
                writer.writerow({'method': name, 'qid': q, 'vid': truth[q]['vid'],
                                 'source_video': truth[q]['vid'].rsplit('_', 2)[0], **row})
    print('Full pairs:', json.dumps(full_pairs), flush=True)
    print('Selected subset:', json.dumps(result['videoarm_success_selected_subset']['metrics']), flush=True)
    print('Subset pair:', json.dumps(pair_partial), flush=True)
    print('MERIT diagnostics:', json.dumps(merit_diagnostics), flush=True)


if __name__ == '__main__':
    main()
