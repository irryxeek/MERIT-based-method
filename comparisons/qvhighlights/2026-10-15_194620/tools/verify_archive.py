from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import gzip
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WRITE_REPORTS = '--write-reports' in sys.argv

def read_jsonl(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write_json(name, value):
    if not WRITE_REPORTS:
        return
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def sha(path, decompress=False):
    h = hashlib.sha256()
    opener = gzip.open if decompress else open
    with opener(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def validate_predictions(rows, manifest, task):
    expected = {row['qid']: row for row in manifest}
    ids = [row['qid'] for row in rows]
    assert len(ids) == len(set(ids)) and set(ids) <= set(expected)
    for row in rows:
        target = expected[row['qid']]
        assert row['vid'] == target['vid'] and row['query'] == target['query']
        if task == 'hd':
            values = row['pred_saliency_scores']
            assert len(values) == int(target['duration'] / 2)
            assert all(finite(value) for value in values)
        else:
            values = row['pred_relevant_windows']
            assert 1 <= len(values) <= 10
            assert all(len(v) == 3 and all(finite(x) for x in v) and 0 <= v[0] < v[1] <= target['duration'] for v in values)
            assert all(a[2] >= b[2] for a,b in zip(values,values[1:]))
    return set(ids)


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if manifest.get('public_export') and '--check-sources' in sys.argv:
        raise SystemExit('--check-sources applies only to the original local archive; run without it for this public export.')
    records = {item['archive_path']:item for item in manifest['files']}
    for item in manifest['files']:
        path = ROOT / item['archive_path']
        assert sha(path) == item['archive_sha256'], item['archive_path']
        expected = item.get('public_uncompressed_sha256', item.get('snapshot_uncompressed_sha256', item['source_sha256']))
        assert sha(path, item['encoding'] == 'gzip') == expected, item['archive_path']
        if '--check-sources' in sys.argv:
            assert sha(Path(item['source'])) == item['source_sha256'], item['source']
    print('Verified file hashes and lossless compression:',len(manifest['files']),flush=True)

    truth = read_jsonl(ROOT / 'common/ground_truth_val.jsonl')
    lookup = {row['qid']:row for row in truth}
    assert len(truth) == len(lookup) == 1550
    assert len({row['vid'] for row in truth}) == 1519
    for folder in ('compact_v1','groundvts_v2b'):
        other = read_jsonl(ROOT / 'longvt' / folder / 'manifests/validation_ground_truth.jsonl')
        assert {row['qid']:row for row in other} == lookup

    sys.path.insert(0,str(ROOT / 'common/evaluator'))
    from standalone_eval.eval import eval_submission

    summary = []
    per_qid = {qid:{'qid':qid,'vid':row['vid'],'query':row['query']} for qid,row in lookup.items()}
    all_native = set(lookup)
    source_hash_checks = []
    for protocol,result_dir in (('compact_v1','full_validation_v1'),('groundvts_v2b','full_v2b')):
        for method in ('global','tool'):
            base = Path('longvt') / protocol / 'results' / result_dir / ('longvt_' + method)
            predictions = read_jsonl(ROOT / base / 'predictions.jsonl')
            assert validate_predictions(predictions,truth,'hd') == set(lookup)
            traces = read_jsonl(ROOT / base / 'traces.jsonl.gz')
            assert len(traces) == len({row['qid'] for row in traces}) == 1550
            assert {row['qid'] for row in traces} == set(lookup)
            pred_map = {row['qid']:row for row in predictions}
            for row in traces:
                assert row['pred_saliency_scores'] == pred_map[row['qid']]['pred_saliency_scores']
                key = 'longvt_' + protocol + '_' + method
                per_qid[row['qid']][key + '_status'] = row['status']
                per_qid[row['qid']][key + '_fallback'] = bool(row['fallback_applied'])
                if row['fallback_applied']:
                    all_native.discard(row['qid'])
            fallback = sum(bool(row['fallback_applied']) for row in traces)
            recorded = json.loads((ROOT / base / 'metrics.json').read_text())
            recomputed = eval_submission(predictions,truth,verbose=False,match_number=True)
            for key,value in recorded.items():
                assert recomputed[key] == value,(str(base),key,recomputed[key],value)
            config = json.loads((ROOT / base / 'config_snapshot.json').read_text())
            for name,expected in config.get('source_sha256',{}).items():
                name = Path(name)
                prefix = '/path/to/workspace/LongVT/reproduction/qvhighlights/'
                relative = str(name).removeprefix(prefix)
                if relative.startswith('groundvts_inference_v1/'):
                    archived = Path('longvt/groundvts_v2b') / relative.removeprefix('groundvts_inference_v1/')
                else:
                    archived = Path('longvt/compact_v1') / relative
                # Compare the retained original identity; public bytes were checked above.
                assert records[str(archived)]['source_sha256'] == expected
                source_hash_checks.append(str(archived))
            row = {
                'project':'LongVT','protocol':protocol,'method':method,'task':'hd','split':'val',
                'planned':1550,'predictions':1550,'native_success':1550-fallback,'failed_fallback':fallback,
                'model':'longvideotool/LongVT-RFT','model_revision':'16ed9314e4ea600dc4682b172a614252f2f383ad',
                'official_metrics':recorded,'metric_scope':'full_validation_with_declared_zero_vector_fallback',
                'prediction_file':str(base/'predictions.jsonl'), 'metrics_file':str(base/'metrics.json'),
            }
            summary.append(row)
            print('Recomputed and matched official metrics:',protocol,method,recorded['HL-min-VeryGood'],flush=True)

    videoarm = json.loads((ROOT / 'videoarm_run_index.json').read_text())
    for run in videoarm:
        base = Path('videoarm/runs') / run['run_id']
        predictions = read_jsonl(ROOT / run['prediction_file'])
        split_manifest = read_jsonl(ROOT / 'videoarm/reproduction/qvhighlights/manifests' / (run['split']+'.jsonl'))
        ids = validate_predictions(predictions,split_manifest,run['task'])
        statuses = read_jsonl(ROOT / base / 'item_statuses.jsonl')
        assert len(statuses) == len({row['qid'] for row in statuses}) == run['planned']
        assert {row['qid'] for row in statuses} == {row['qid'] for row in split_manifest}
        assert dict(Counter(row['status'] for row in statuses)) == run['statuses']
        assert ids == {row['qid'] for row in statuses if row['status']=='succeeded'}
        scores_path = ROOT / base / 'scores.json'
        if scores_path.exists():
            scores = json.loads(scores_path.read_text())
            assert scores['predictions_sha256'] == sha(ROOT / run['prediction_file'])
            assert scores['manifest_sha256'] == records['videoarm/reproduction/qvhighlights/manifests/' + run['split'] + '.jsonl']['source_sha256']
        if run['split']=='val':
            assert {r['qid']: (r['vid'],r['query'],r['duration']) for r in split_manifest} == {qid:(r['vid'],r['query'],r['duration']) for qid,r in lookup.items()}
            for row in statuses:
                per_qid[row['qid']]['videoarm_' + run['task'] + '_status'] = row['status']
            summary.append({**run,'metric_scope':run['official_score_scope']})
            if run['task']=='hd':
                hd_ids = ids
    common = sorted(hd_ids)
    common_native = sorted(hd_ids & all_native)
    write_json('alignment/common_available_hd_qids.json',common)
    write_json('alignment/common_native_success_hd_qids.json',common_native)
    if WRITE_REPORTS:
        with (ROOT / 'alignment/query_alignment.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(per_qid[min(per_qid)]))
            writer.writeheader()
            writer.writerows(per_qid[qid] for qid in sorted(per_qid))
    write_json('comparison_summary.json',{
        'dataset':'QVHighlights','split':'validation','planned_queries':1550,'unique_videos':1519,
        'official_evaluator_commit':manifest['evaluator_commit'],'runs':summary,
        'all_validation_qids_and_bindings_match':True,
        'common_available_hd_prediction_count':len(common),
        'common_native_success_hd_count':len(common_native),
        'comparability':'Same validation examples and official HD definitions; models, observations, prompts and failure policies differ. VideoARM HD is partial and is not a full-validation result; MR metrics are not HD metrics.',
    })
    if WRITE_REPORTS:
        with (ROOT / 'comparison_summary.csv').open('w',newline='') as stream:
            fields=['project','protocol','method','task','split','planned','predictions','native_success','failed','paused','pending','VeryGood_mAP','VeryGood_Hit1','metric_scope']
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for run in summary:
                vg=run.get('official_metrics',{}).get('HL-min-VeryGood',{}) if run.get('official_metrics') else {}
                status=run.get('statuses',{})
                writer.writerow({
                    **{k:run.get(k,'') for k in fields if k in run},
                    'native_success':run.get('native_success',status.get('succeeded',0)),
                    'failed':run.get('failed_fallback',status.get('failed',0)),
                    'paused':status.get('paused',0),'pending':status.get('pending',0),
                    'VeryGood_mAP':vg.get('HL-mAP',''),'VeryGood_Hit1':vg.get('HL-Hit1',''),
                })
    write_json('verification.json',{
        'verified_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'source_artifacts_verified':len(manifest['files']),'source_files_rechecked':'--check-sources' in sys.argv,
        'longvt_complete_prediction_runs':4,'longvt_predictions':6200,'longvt_metrics_recomputed_and_matched':4,
        'longvt_frozen_source_files_checked':sorted(set(source_hash_checks)),
        'videoarm_runs_validated':len(videoarm),'videoarm_sqlite_backups_quick_check':'all_ok_at_backup',
        'common_available_hd_predictions':len(common),'common_native_success_hd_predictions':len(common_native),
        'annotation_semantic_equality':True,
        'limitations':['VideoARM HD full-validation score is unavailable; no new partial score is presented.',
                       'No new inference, training, selected reruns, or resumption of paused jobs.',
                       'Current MERIT predictions were not copied or scored while the main experiment is active.'],
    })
    print(json.dumps({'status':'verified','common_hd_predictions':len(common),'common_native_hd_predictions':len(common_native),'videoarm_runs':len(videoarm)}),flush=True)

if __name__=='__main__':
    main()
