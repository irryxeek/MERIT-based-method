"""Read frozen timing ledgers without inference or changes to source artifacts."""

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

OUTPUT = Path(__file__).resolve().parent
ARCHIVE = OUTPUT.parent / '2026-10-15_194620'
BASE = ARCHIVE / 'videoarm/runs/formal_c0c8ae9799ad_hd_val_M1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--merit-run', type=Path, required=True,
                    help='Original MERIT run with queries, memories, requests and execution logs')
MERIT = parser.parse_args().merit_run.resolve()


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), sum=float(a.sum()))


with tempfile.TemporaryDirectory(prefix='videoarm-timing-') as directory:
    database = Path(directory) / 'state.sqlite'
    with gzip.open(BASE / 'state.sqlite.gz', 'rb') as stream:
        database.write_bytes(stream.read())
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    items = {r['qid']: {**dict(r), 'budget': json.loads(r['budget'])}
             for r in connection.execute('SELECT * FROM items')}
    requests = [dict(r) for r in connection.execute('SELECT qid,stage,elapsed FROM requests')]
    connection.close()

shared = [q for q, r in items.items() if r['status'] == 'succeeded']
merit_queries = {q: json.loads((MERIT / f'queries/{q}.json').read_text()) for q in shared}
native = [q for q in shared if merit_queries[q]['status'] == 'ok']
memories = {merit_queries[q]['vid']: json.loads(
    (MERIT / 'memories' / f"{merit_queries[q]['vid']}.json").read_text()) for q in native}
cache = {}


def request_totals(events):
    seconds = tokens = attempts = 0
    for rid in {e['request_id'] for e in events}:
        if rid not in cache:
            cache[rid] = json.loads((MERIT / 'requests' / rid[:2] / (rid + '.json')).read_text())
        for attempt in cache[rid]['attempts']:
            seconds += attempt['seconds']
            tokens += attempt.get('response', {}).get('usage', {}).get('total_tokens', 0)
            attempts += 1
    return dict(seconds=seconds, tokens=tokens, attempts=attempts)


per_query = []
for q in native:
    query = merit_queries[q]
    memory = memories[query['vid']]
    budget = items[q]['budget']
    totals = request_totals(query['requests'] + memory['requests'])
    per_query.append(dict(
        qid=q, vid=query['vid'], merit_memory_seconds=memory['seconds'],
        merit_query_seconds=query['seconds'],
        merit_memory_plus_query_seconds=memory['seconds'] + query['seconds'],
        merit_vlm_request_seconds=totals['seconds'], merit_tokens=totals['tokens'],
        merit_request_attempts=totals['attempts'],
        merit_cache_hits=sum(e['cache_hit'] for e in memory['requests'] + query['requests']),
        videoarm_active_seconds=budget['active_seconds'],
        videoarm_request_elapsed=sum(r['elapsed'] or 0 for r in requests if r['qid'] == q),
        videoarm_tokens=budget['actual_input_tokens'] + budget['actual_output_tokens'],
        videoarm_requests=budget['actual_requests'],
        videoarm_shared_cache_hits=budget['shared_cache_hits'],
    ))
summary = {k: stats([r[k] for r in per_query]) for k in per_query[0] if k not in ('qid', 'vid')}
execution = json.loads((BASE / 'execution.json').read_text())
previous_path = next((MERIT / 'runtime').glob('execution_previous_*.json'))
previous = json.loads(previous_path.read_text())
finished = json.loads((MERIT / 'execution_status.json').read_text())
merit_wall = finished['updated_at'] - previous['started_at']
videoarm_wall = execution['total_wall_seconds']
result = {
    'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'common_success_queries': len(native), 'common_success_videos': len(memories),
    'paired_statistics': summary,
    'videoarm_all_success_active_seconds': stats([items[q]['budget']['active_seconds'] for q in shared]),
    'videoarm_statuses': dict(Counter(r['status'] for r in items.values())),
    'videoarm_all_budget_totals': {k: sum(r['budget'].get(k, 0) for r in items.values()) for k in [
        'requests', 'actual_requests', 'cache_hits', 'shared_cache_hits',
        'actual_input_tokens', 'actual_output_tokens', 'active_seconds']},
    'observed_execution': {
        'MERIT': {
            'wall_seconds': merit_wall, 'completed_queries': 1550, 'successful_queries': 1541,
            'workers': 1, 'completed_queries_per_hour': 1550 / merit_wall * 3600,
            'successful_queries_per_hour': 1541 / merit_wall * 3600,
            'scope': 'Recorded main launch to supervisor completion; includes smoke/restart/evaluation, excludes earlier model/frame preparation and previously prepared memory',
        },
        'VideoARM': {
            'wall_seconds': videoarm_wall, 'terminal_queries': 453, 'successful_queries': 419,
            'partially_paused_queries': 4, 'workers': 4,
            'terminal_queries_per_hour': 453 / videoarm_wall * 3600,
            'successful_queries_per_hour': 419 / videoarm_wall * 3600,
            'scope': 'Recorded partial validation runner wall time to pause; includes failure/paused work, excludes service startup',
        },
    },
    'notes': [
        'MERIT memory+query timing excludes frame preparation/index/overhead and charges full recorded memory build per query even when videos repeat; VideoARM active_seconds includes intra-query work and concurrent service waits.',
        'MERIT per-request timings include original records of any reused responses; cache counts are retained.',
        'Different service versions, concurrency, pixels and scopes: not a controlled speed benchmark.',
        'Query latency sums in a concurrent run are not GPU time.',
    ],
    'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        BASE / 'state.sqlite.gz', BASE / 'execution.json', MERIT / 'execution_status.json', previous_path]},
}
(OUTPUT / 'timing_comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
with (OUTPUT / 'timing_per_query.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(per_query[0]))
    writer.writeheader()
    writer.writerows(per_query)
print(json.dumps(result, indent=2, ensure_ascii=False))
