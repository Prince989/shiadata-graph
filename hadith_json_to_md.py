"""Concatenate phase1 hadith JSON files into a single markdown document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.paths import OUTPUT_DIR

HADITH_DIR = OUTPUT_DIR / "phase1" / "hadith"
DEFAULT_OUT = OUTPUT_DIR / "phase1" / "hadith.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read hadith JSON files and write them into one markdown file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=HADITH_DIR,
        help=f"Directory of JSON files (default: {HADITH_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Markdown output path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files in {input_dir}")

    blocks: list[str] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        blocks.append(f"[ {path.name} ]\n{pretty}")

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} file(s) to {output}")


if __name__ == "__main__":
    main()
