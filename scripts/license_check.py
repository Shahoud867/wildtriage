"""List models/datasets that are not cleared for commercial use (from ml/models.yaml).

    python scripts/license_check.py                 # report
    python scripts/license_check.py --commercial    # exit 1 if anything in use is not cleared
"""

import argparse
import sys
from pathlib import Path

import yaml

MANIFEST = Path(__file__).resolve().parents[1] / "ml" / "models.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commercial", action="store_true", help="fail if any entry is not commercial_ok: true")
    parser.add_argument("--only", nargs="*", help="names to check (default: all)")
    args = parser.parse_args()
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    blocked = []
    for kind in ("models", "datasets"):
        for entry in manifest.get(kind, []):
            if args.only and entry["name"] not in args.only:
                continue
            status = entry.get("commercial_ok")
            mark = "ok" if status is True else ("NO" if status is False else "VERIFY")
            print(f"{kind[:-1]:8s} {entry['name']:28s} {mark:7s} {entry.get('license', '?')}")
            if status is not True:
                blocked.append(entry["name"])
    if args.commercial and blocked:
        print(f"\nNot cleared for commercial use: {', '.join(blocked)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
