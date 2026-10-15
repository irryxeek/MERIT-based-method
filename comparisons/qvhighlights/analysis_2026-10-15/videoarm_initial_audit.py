"""Rescore archived VideoARM initialization without inference or archive changes."""

import csv
import gzip
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

from analyze import cluster_comparison, read_jsonl, sha256


def main():
    output = Path(__file__).resolve().parent
    archive = output.parent / '2026-10-15_194620'
    run = archive / 'videoarm/runs/formal_c0c8ae9799ad_hd_val_M1'
    sys.path.insert(0, str(archive / 'common/evaluator'))
    from standalone_eval.eval import compute_ap_from_tuple, compute_hl_ap, compute_hl_hit1, mk_gt_scores

    inputs = [run / 'state.sqlite.gz', run / 'predictions.jsonl',
              archive / 'common/ground_truth_val.jsonl', output / 'analysis.json',
              output / 'analyze.py']
    inputs += [archive / 'common/evaluator/standalone_eval' / n for n in ['eval.py', 'utils.py']]
    hashes = {str(p.relative_to(output.parent)): sha256(p) for p in inputs}
    truth = {r['qid']: r for r in read_jsonl(inputs[2])}
    exported = {r['qid']: r for r in read_jsonl(inputs[1])}
    initial, final, changed = {}, {}, {}
    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / 'audit.sqlite'
        db_path.write_bytes(gzip.decompress(inputs[0].read_bytes()))
        connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        try:
            assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            for qid, snapshot_text, prediction_text in connection.execute(
                    "SELECT qid,snapshot,prediction FROM items WHERE status='succeeded'"):
                snapshot, prediction = json.loads(snapshot_text), json.loads(prediction_text)
                assert prediction == exported[qid]
                assert prediction['vid'] == truth[qid]['vid']
                entries = [entry for batch in snapshot['initial_batches'] for entry in batch['clip_scores']]
                scores = {entry['clip_id']: entry['score'] for entry in entries}
                n = int(truth[qid]['duration'] / 2)
                assert len(entries) == len(scores) == n
                assert set(scores) == set(range(n))
                vector = np.array([scores[i] for i in range(n)], dtype=float)
                final_vector = np.array(prediction['pred_saliency_scores'], dtype=float)
                assert len(final_vector) == n
                for values in [vector, final_vector]:
                    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 4).all()
                initial[qid] = {**prediction, 'pred_saliency_scores': vector.tolist()}
                final[qid] = prediction
                changed[qid] = int(np.count_nonzero(vector != final_vector))
        finally:
            connection.close()
    assert set(initial) == set(final) == set(exported) and len(final) == 419

    thresholds = [('Fair', 2), ('Good', 3), ('VeryGood', 4)]
    columns = [f'{name}_{metric}' for name, _ in thresholds for metric in ['mAP', 'Hit1']]
    per_query, metrics = {}, {}
    for name, predictions in [('initial', initial), ('final', final)]:
        per_query[name] = {q: {} for q in predictions}
        metrics[name] = {}
        for label, threshold in thresholds:
            binary = {q: (mk_gt_scores(truth[q]) >= threshold).astype(float) for q in predictions}
            for q, prediction in predictions.items():
                scores = np.asarray(prediction['pred_saliency_scores'])
                ap = [compute_ap_from_tuple((0, w, binary[q][:, w], scores))[2] for w in range(3)]
                per_query[name][q][f'{label}_mAP'] = float(100 * np.mean(ap))
                per_query[name][q][f'{label}_Hit1'] = float(100 * binary[q][int(np.argmax(scores))].max())
            metrics[name][f'{label}_mAP'] = compute_hl_ap(predictions, binary, num_workers=1)
            metrics[name][f'{label}_Hit1'] = compute_hl_hit1(predictions, binary)
        for column in columns:
            assert round(np.mean([r[column] for r in per_query[name].values()]), 2) == metrics[name][column]

    previous = json.loads((output / 'analysis.json').read_text())
    native_qids = set(previous['merit_videoarm_pair_native_success']['qids'])
    assert len(native_qids) == 413 and native_qids <= set(final)
    expected = previous['videoarm_success_selected_subset']['metrics']['VideoARM_HD_partial']
    for column in columns:
        assert abs(np.mean([r[column] for r in per_query['final'].values()]) - expected[column]) < 1e-9
    result = {
        'scope': 'Post-hoc rescoring of archived initialization and final outputs; no model calls',
        'warning': '419 final-success-selected queries; not a full baseline run or an unselected ablation',
        'queries': len(final), 'unchanged_score_vectors': sum(n == 0 for n in changed.values()),
        'changed_score_vectors': sum(n > 0 for n in changed.values()),
        'changed_clip_scores': sum(changed.values()), 'metrics': metrics,
        'final_minus_initial': cluster_comparison(per_query['final'], per_query['initial'], truth, columns),
        'merit_shared_native_413': {name: {c: float(np.mean([values[q][c] for q in native_qids]))
                                        for c in columns} for name, values in per_query.items()},
        'bootstrap': previous['bootstrap'], 'input_sha256': hashes,
        'script_sha256': sha256(Path(__file__)),
    }
    assert hashes == {str(p.relative_to(output.parent)): sha256(p) for p in inputs}
    (output / 'videoarm_initial_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (output / 'videoarm_initial_per_query.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['qid', 'vid', 'changed_clip_scores'] +
                                [f'{name}_{c}' for name in ['initial', 'final'] for c in columns])
        writer.writeheader()
        for q in sorted(final):
            writer.writerow({'qid': q, 'vid': truth[q]['vid'], 'changed_clip_scores': changed[q],
                             **{f'{name}_{c}': per_query[name][q][c]
                                for name in ['initial', 'final'] for c in columns}})
    print(json.dumps({k: v for k, v in result.items() if k not in ['input_sha256', 'script_sha256']}, indent=2))


if __name__ == '__main__':
    main()
