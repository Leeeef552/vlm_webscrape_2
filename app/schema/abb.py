#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict
from typing import Union, Iterable

def ensure_list(v: Union[str, Iterable[str]]) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)

def merge_json_folder(src_dir: str, pattern: str = "*.json") -> dict[str, list[str]]:
    merged: dict[str, list[str]] = defaultdict(list)

    for fp in Path(src_dir).glob(pattern):
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for abbr, meanings in data.items():
            for m in ensure_list(meanings):
                if m not in merged[abbr]:
                    merged[abbr].append(m)

    return dict(merged)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Merge abbreviation JSON files.")
    p.add_argument("folder", help="Folder containing JSON files")
    p.add_argument("-o", "--out", default="merged.json", help="Output file (default: merged.json)")
    p.add_argument("--pattern", default="*.json", help="Glob pattern to match files (default: *.json)")
    args = p.parse_args()

    result = merge_json_folder(args.folder, args.pattern)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Written {len(result)} abbreviations to {args.out}")
