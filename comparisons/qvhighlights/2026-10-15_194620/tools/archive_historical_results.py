from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile

PROJECT = Path('/path/to/workspace/MERIT')
LONGVT = Path('/path/to/workspace/LongVT/reproduction/qvhighlights')
GROUNDVTS = LONGVT / 'groundvts_inference_v1'
VIDEOARM = Path('/path/to/storage/videoarm-repro')
QVH = VIDEOARM / 'reproduction/qvhighlights'
RUNS = VIDEOARM / 'qvh_data/runs'
EVALUATOR = Path('/path/to/storage/longvt-repro/benchmark_sources/moment_detr')
STAMP = datetime.now(ZoneInfo('Asia/Shanghai'))
ARCHIVE = PROJECT / 'comparisons/qvhighlights' / STAMP.strftime('%Y-%m-%d_%H%M%S')
ARCHIVE.mkdir(parents=True, exist_ok=False)
Path('/path/to/temporary/merit_comparison_archive_path.txt').write_text(str(ARCHIVE))
entries = []
runs = []
secret_patterns = [
    re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED)? ?PRIVATE KEY-----'),
    re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})'),
    re.compile(rb'hf_[A-Za-z0-9]{25,}'),
]

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def guard(data, name):
    if any(pattern.search(data) for pattern in secret_patterns):
        raise RuntimeError(f'Credential pattern detected; refusing to archive {name}')

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def copy_file(source, relative, compress=False, provenance=None):
    source = Path(source)
    if source.is_symlink():
        raise RuntimeError(f'Unexpected file symlink: {source}')
    original_stat = source.stat()
    data = source.read_bytes()
    guard(data, str(source))
    original_hash = sha_bytes(data)
    destination = ARCHIVE / (str(relative) + ('.gz' if compress else ''))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f'Duplicate destination: {destination}')
    if compress:
        with destination.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as stream:
                stream.write(data)
    else:
        destination.write_bytes(data)
    if file_sha(source) != original_hash:
        raise RuntimeError(f'Source changed during snapshot: {source}')
    record = {
        'source': str(source.resolve()), 'archive_path': str(destination.relative_to(ARCHIVE)),
        'source_sha256': original_hash, 'source_bytes': len(data),
        'source_mtime_ns': original_stat.st_mtime_ns,
        'archive_sha256': file_sha(destination), 'archive_bytes': destination.stat().st_size,
        'encoding': 'gzip' if compress else 'identity', 'kind': 'byte_preserving_copy',
    }
    if provenance:
        record.update(provenance)
    entries.append(record)

def copy_tree(source, relative, suffixes, exclude_dirs=(), compress_names=()):
    source = Path(source)
    for folder, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in set(exclude_dirs) | {'__pycache__', '.git', '.cache'})
        for name in sorted(files):
            p = Path(folder) / name
            if p.suffix in suffixes or name in ('COMPLETED',):
                copy_file(p, Path(relative) / p.relative_to(source), compress=name in compress_names)

def sqlite_snapshot(source, relative):
    before = file_sha(source)
    with tempfile.TemporaryDirectory(prefix='merit-ledger-') as temporary:
        temporary_db = Path(temporary) / 'state.sqlite'
        readonly = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
        target = sqlite3.connect(temporary_db)
        try:
            readonly.backup(target)
            target.commit()
            assert target.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
            statuses = dict(target.execute('SELECT status, COUNT(*) FROM items GROUP BY status').fetchall())
            items = [dict(zip(('qid', 'status', 'error', 'updated'), row)) for row in target.execute('SELECT qid,status,error,updated FROM items ORDER BY qid')]
            predictions = [json.loads(row[0]) for row in target.execute('SELECT prediction FROM items WHERE prediction IS NOT NULL ORDER BY qid')]
            requests = dict(target.execute('SELECT status,COUNT(*) FROM requests GROUP BY status').fetchall())
        finally:
            target.close()
            readonly.close()
        assert file_sha(source) == before, f'Source database changed: {source}'
        copy_file(temporary_db, relative, compress=True, provenance={
            'source': str(source), 'source_sha256': before,
            'kind': 'consistent_sqlite_backup', 'snapshot_uncompressed_sha256': file_sha(temporary_db),
            'snapshot_uncompressed_bytes': temporary_db.stat().st_size,
            'source_bytes': source.stat().st_size, 'source_mtime_ns': source.stat().st_mtime_ns,
            'sqlite_quick_check': 'ok',
        })
    return statuses, items, predictions, requests

# Complete LongVT HD experiments, preserving the source layout for small reports.
for origin, destination, report_names, result_subdir in (
    (LONGVT, Path('longvt/compact_v1'), ['README.md', 'FULL_VALIDATION_REPORT.md', 'ADAPTATION_REPORT_ZH.md'], 'results/full_validation_v1'),
    (GROUNDVTS, Path('longvt/groundvts_v2b'), ['README.md', 'FINAL_REPORT_ZH.md', 'PROGRESS.md', 'PLAN.md'], 'results/full_v2b'),
):
    for name in report_names:
        if (origin / name).exists():
            copy_file(origin / name, destination / name)
    for directory in ('configs', 'scripts', 'manifests'):
        copy_tree(origin / directory, destination / directory, {'.json', '.jsonl', '.py', '.sh', '.txt'})
    copy_tree(origin / result_subdir, destination / result_subdir, {'.json', '.jsonl', '.csv', '.txt', '.log'},
              exclude_dirs=('local_frames', 'frames'), compress_names=('traces.jsonl', 'attempts.jsonl'))
    print('Archived', destination, flush=True)
copy_tree(GROUNDVTS / 'analysis/full_v2b', Path('longvt/groundvts_v2b/analysis/full_v2b'), {'.json', '.jsonl', '.csv'})

# VideoARM reports, protocol and implementation; never copy .env, environments, media, or models.
for path in sorted(QVH.glob('*.md')) + [QVH / 'progress.json']:
    copy_file(path, Path('videoarm/reproduction/qvhighlights') / path.name)
for directory in ('configs', 'manifests', 'tools'):
    copy_tree(QVH / directory, Path('videoarm/reproduction/qvhighlights') / directory, {'.json', '.jsonl', '.py', '.lock'})
for path in sorted((QVH / 'checks').iterdir()):
    if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.txt', '.md'):
        continue
    if path.name.startswith(('gpu_', 'disks_', 'svh_', 'protected_artifacts_')):
        continue
    copy_file(path, Path('videoarm/reproduction/qvhighlights/checks') / path.name)
copy_tree(VIDEOARM / 'open_source_qvh', Path('videoarm/open_source_qvh'), {'.py', '.md', '.json'})
copy_tree(QVH / 'archive', Path('videoarm/reproduction/qvhighlights/archive'), {'.py', '.json', '.md', '.txt'})

for run in sorted(RUNS.iterdir()):
    if not (run / 'state.sqlite').is_file():
        continue
    relative = Path('videoarm/runs') / run.name
    identity = json.loads((run / 'identity.json').read_text())
    config = identity['config']
    for path in sorted(run.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.jsonl'):
            copy_file(path, relative / path.name)
    statuses, items, predictions, requests = sqlite_snapshot(run / 'state.sqlite', relative / 'state.sqlite')
    # Export readable statuses independently of the compressed request ledger.
    item_path = ARCHIVE / relative / 'item_statuses.jsonl'
    item_path.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items))
    on_disk = [json.loads(line) for line in (run / 'predictions.jsonl').read_text().splitlines() if line]
    assert len(on_disk) == len({row['qid'] for row in on_disk})
    assert {r['qid']: r for r in on_disk} == {r['qid']: r for r in predictions}
    assert len(on_disk) == statuses.get('succeeded', 0)
    scores_path = run / 'scores.json'
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else None
    runs.append({
        'project': 'VideoARM', 'run_id': run.name, 'task': config['task'],
        'method': config['method'], 'split': config['split'],
        'protocol': config['protocol'], 'model': config['model'], 'model_revision': config['model_revision'],
        'planned': len(items), 'statuses': statuses, 'predictions': len(on_disk),
        'requests': requests, 'prediction_file': str(relative / 'predictions.jsonl'),
        'state_snapshot': str(relative / 'state.sqlite.gz'),
        'scores_file': str(relative / 'scores.json') if scores else None,
        'official_score_scope': scores.get('status') if scores else 'not_scored',
        'official_metrics': scores.get('official_metrics', {}).get('brief') if scores else None,
        'custom_full_plan_failure_zero_metrics': scores.get('custom_full_plan_failure_zero_metrics') if scores else None,
        'historical_contract_issue': 'v1' in run.name,
    })
    print('Archived', run.name, 'predictions', len(on_disk), flush=True)

# Validation annotations and the exact official scoring implementation for offline comparisons.
copy_file(EVALUATOR / 'data/highlight_val_release.jsonl', Path('common/ground_truth_val.jsonl'))
copy_tree(EVALUATOR / 'standalone_eval', Path('common/evaluator/standalone_eval'), {'.py', '.md'})
for name in ('LICENSE', 'README.md'):
    if (EVALUATOR / name).exists():
        copy_file(EVALUATOR / name, Path('common/evaluator') / name)
train = {r['qid']: r for r in map(json.loads, (EVALUATOR / 'data/highlight_train_release.jsonl').read_text().splitlines())}
dev_qids = set()
for name in ('smoke8.jsonl', 'dev50.jsonl'):
    dev_qids.update(json.loads(line)['qid'] for line in (QVH / 'manifests' / name).read_text().splitlines())
(ARCHIVE / 'common/ground_truth_development.jsonl').write_text(''.join(json.dumps(train[q], ensure_ascii=False) + '\n' for q in sorted(dev_qids)))

manifest = {
    'created_at': STAMP.isoformat(timespec='seconds'), 'scope': 'Historical QVHighlights experiments for MERIT comparison; local archive',
    'evaluator_commit': 'b7e553ac3b0c898ee6b85e03ee507c064eab89ca',
    'files': entries,
    'exclusions': [
        'Model checkpoints, source videos, sampled image files, virtual environments, credentials and .env files',
        'VideoARM per-stage trace files duplicating the SQLite request/response ledger; checkpoint states remain in SQLite items',
        'Unrelated VideoSIAH/SVHighlights experiments and incomplete LongVT LoRA training',
        'LongVT developmental pilot/smoke prediction runs; both completed full-validation protocols are retained',
        'Test-set labels; validation labels and selected development labels only',
    ],
    'compression_note': 'traces.jsonl and attempts.jsonl use lossless gzip; SQLite ledgers use consistent backup then gzip. Original sources are retained.',
    'report_note': 'Historical reports/configs are copied verbatim, including original paths and dated statements. They are archival evidence, not instructions to restart experiments.',
}
write_json(ARCHIVE / 'manifest.json', manifest)
write_json(ARCHIVE / 'videoarm_run_index.json', runs)
print(json.dumps({'archive': str(ARCHIVE), 'source_artifacts':len(entries), 'videoarm_runs':len(runs), 'archive_bytes':sum(p.stat().st_size for p in ARCHIVE.rglob('*') if p.is_file())}), flush=True)
