"""Daily Phase 1 run report: per-key HTTP counts + flushed hadith ids.

Written under `data/reports/phase1/YYYY-MM-DD.md` (plus a `.json` sidecar so
same-day re-runs merge instead of overwriting). The directory is gitignored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from config.paths import DATA_DIR

REPORT_DIR = DATA_DIR / "reports" / "phase1"


def _clean_hadith_id(label: str) -> str:
    """Tidy older ids that still contain Persian تا."""
    return re.sub(r"_تا_", "-", str(label or "").strip())


def _safe_token(value: str, *, keep_hyphen: bool = False) -> str:
    allowed = r"\w\u0600-\u06FF\-" if keep_hyphen else r"\w\u0600-\u06FF"
    text = re.sub(rf"[^{allowed}]+", "_", str(value or "").strip(), flags=re.UNICODE)
    return re.sub(r"_+", "_", text).strip("_") or "x"


def _page_token(locator: str, page_start: str = "") -> str:
    """Normalize `صفحه 20 تا 23` → `20-23` (never keep Persian تا in the id)."""
    source = locator or page_start
    match = re.search(
        r"صفحه\s*([0-9\u06F0-\u06F9]+)(?:\s*تا\s*([0-9\u06F0-\u06F9]+))?",
        source,
    )
    if not match:
        return "x"
    start = match.group(1)
    end = match.group(2)
    return f"{start}-{end}" if end and end != start else start


def hadith_run_id(source_path: str, payload: dict) -> str:
    """bookname_vol_page_marker, e.g. al-kafi-1_1_20_13 or al-kafi-1_1_20-23_14."""
    book = _safe_token(Path(source_path).stem or "unknown", keep_hyphen=True)
    locator = str(payload.get("locator") or "")
    vol_m = re.search(r"جلد\s*(\d+)", locator)
    vol = vol_m.group(1) if vol_m else "x"
    page = _page_token(locator, str(payload.get("page_start") or ""))
    marker = _safe_token(payload.get("marker") or "x")
    return f"{book}_{vol}_{page}_{marker}"


def _pad(text: str, width: int, *, align: str = "left") -> str:
    value = str(text)
    if len(value) >= width:
        return value
    pad = width - len(value)
    if align == "right":
        return (" " * pad) + value
    if align == "center":
        left = pad // 2
        return (" " * left) + value + (" " * (pad - left))
    return value + (" " * pad)


def _ascii_table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> list[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    head = (
        "| "
        + " | ".join(_pad(h, widths[i], align="center") for i, h in enumerate(headers))
        + " |"
    )
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                _pad(cell, widths[i], align=aligns[i]) for i, cell in enumerate(row)
            )
            + " |"
        )
    return [sep, head, sep, *body, sep]


@dataclass
class _KeyStats:
    count_200: int = 0
    count_503: int = 0
    count_429: int = 0
    hadiths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "200": self.count_200,
            "503": self.count_503,
            "429": self.count_429,
            "hadiths": list(self.hadiths),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "_KeyStats":
        data = data or {}
        return cls(
            count_200=int(data.get("200") or 0),
            count_503=int(data.get("503") or 0),
            count_429=int(data.get("429") or 0),
            hadiths=[str(x) for x in (data.get("hadiths") or []) if str(x).strip()],
        )


class Phase1DayReport:
    """Accumulate one calendar day's Phase 1 Gemini + flush stats.

    Writes the markdown (and JSON sidecar) after every event so a cancel,
    crash, or quota stop still leaves a usable report on disk.
    """

    def __init__(
        self,
        day: date | None = None,
        report_dir: Path | None = None,
        *,
        live: bool = True,
    ):
        self.day = day or date.today()
        self.report_dir = report_dir or REPORT_DIR
        self.live = live
        self._keys: dict[int, _KeyStats] = {}
        self._last_ok_key: int | None = None
        self._load()
        # Create the day file immediately so the directory is visible mid-run.
        if self.live:
            self.flush()

    @property
    def json_path(self) -> Path:
        return self.report_dir / f"{self.day.isoformat()}.json"

    @property
    def md_path(self) -> Path:
        return self.report_dir / f"{self.day.isoformat()}.md"

    def _key(self, key_no: int) -> _KeyStats:
        # Display as 1-based key number (matches "Using Gemini key 1/4").
        n = int(key_no)
        if n not in self._keys:
            self._keys[n] = _KeyStats()
        return self._keys[n]

    def record_http(self, key_index: int, status: int) -> None:
        """`key_index` is 0-based from LlmKey.index; stored as key no = index+1."""
        row = self._key(key_index + 1)
        if status == 200:
            row.count_200 += 1
            self._last_ok_key = key_index + 1
        elif status == 503:
            row.count_503 += 1
        elif status == 429:
            row.count_429 += 1
        if self.live:
            self.flush()

    def record_hadith(self, source_path: str, payload: dict, key_no: int | None = None) -> None:
        label = hadith_run_id(source_path, payload)
        target = key_no if key_no is not None else self._last_ok_key
        if target is None:
            target = 0
            row = self._key(0)
        else:
            row = self._key(target)
        if label not in row.hadiths:
            row.hadiths.append(label)
        if self.live:
            self.flush()

    def _load(self) -> None:
        path = self.json_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for key, stats in (data.get("keys") or {}).items():
            try:
                self._keys[int(key)] = _KeyStats.from_dict(stats)
            except (TypeError, ValueError):
                continue

    def flush(self) -> Path:
        """Rewrite JSON + markdown from current in-memory totals."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "date": self.day.isoformat(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "keys": {str(k): v.to_dict() for k, v in sorted(self._keys.items())},
        }
        self.json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.md_path.write_text(self.render_markdown(payload), encoding="utf-8")
        return self.md_path

    @staticmethod
    def render_markdown(payload: dict) -> str:
        day = payload.get("date") or ""
        updated = payload.get("updated_at") or ""
        keys = payload.get("keys") or {}

        headers = ["key", "503", "200", "429", "hadiths"]
        aligns = ["right", "right", "right", "right", "right"]
        rows: list[list[str]] = []
        hadith_sections: list[str] = []

        if not keys:
            rows.append(["—", "0", "0", "0", "0"])
        else:
            for key in sorted(keys, key=lambda k: int(k)):
                stats = keys[key] or {}
                label = "?" if str(key) == "0" else str(key)
                hadiths = [
                    _clean_hadith_id(h)
                    for h in (stats.get("hadiths") or [])
                    if str(h).strip()
                ]
                rows.append(
                    [
                        label,
                        str(int(stats.get("503") or 0)),
                        str(int(stats.get("200") or 0)),
                        str(int(stats.get("429") or 0)),
                        str(len(hadiths)),
                    ]
                )
                if hadiths:
                    hadith_sections.append(f"### Key {label}")
                    hadith_sections.append("")
                    for item in hadiths:
                        hadith_sections.append(f"- `{item}`")
                    hadith_sections.append("")

        lines = [
            f"# Phase 1 - {day}",
            "",
            f"Updated `{updated}`",
            "",
            "```",
            *_ascii_table(headers, rows, aligns),
            "```",
            "",
        ]
        if hadith_sections:
            lines.append("## Processed hadiths")
            lines.append("")
            lines.extend(hadith_sections)
        else:
            lines.append("_No hadiths flushed yet._")
            lines.append("")
        return "\n".join(lines)
