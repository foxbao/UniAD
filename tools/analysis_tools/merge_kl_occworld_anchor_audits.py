#!/usr/bin/env python
"""Merge disjoint predicted-box anchor-audit shards."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis_tools.audit_kl_occworld_predicted_box_anchor import (  # noqa: E402
    _aggregate,
)


def merge_reports(paths: list) -> dict:
    if not paths:
        raise ValueError('Provide at least one anchor-audit report')
    reports = []
    for path in paths:
        with Path(path).open() as source:
            reports.append(json.load(source))
    box_sources = {report.get('box_source') for report in reports}
    if len(box_sources) != 1:
        raise ValueError(f'Anchor reports mix box sources: {box_sources}')
    rows = [row for report in reports for row in report.get('rows', [])]
    references = [int(row['reference_index']) for row in rows]
    if not rows or len(references) != len(set(references)):
        raise ValueError('Anchor reports are empty or contain duplicate references')
    rows.sort(key=lambda row: int(row['reference_index']))
    return {
        'schema_version': 1,
        'purpose': (
            'Merged current-frame box-source substitution audit. This is '
            'not a full historical predicted-box or final-holdout evaluation.'),
        'box_source': box_sources.pop(),
        'reference_count': len(rows),
        'source_reports': [str(Path(path)) for path in paths],
        'aggregate': _aggregate(rows),
        'rows': rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--out-file', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = merge_reports(args.reports)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
