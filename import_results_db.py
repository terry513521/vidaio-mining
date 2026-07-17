#!/usr/bin/env python3
"""Import existing fleet result.json files into the SQLite results DB."""

from __future__ import annotations

import argparse
import json

from results_db import DEFAULT_DB_PATH, import_work_root, list_latest_results


def main() -> int:
    p = argparse.ArgumentParser(description="Import fleet results into SQLite")
    p.add_argument("--work-root", default="work_fleet")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p.add_argument("--list", action="store_true", help="List latest rows after import")
    args = p.parse_args()

    run_id, count = import_work_root(args.work_root, db_path=args.db)
    print(f"imported={count} run_id={run_id} db={args.db}")
    if args.list:
        rows = list_latest_results(db_path=args.db, limit=20)
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
