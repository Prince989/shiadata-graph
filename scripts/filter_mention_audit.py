"""Hybrid mention-audit filter: noisy deterministic candidates → LLM FP drop.

Completely offline from phase-1 extraction. The system prompt below is local
to this script and MUST NOT be imported from (or merged with) phase-1 prompts.

Example:
    python scripts/filter_mention_audit.py
    python scripts/filter_mention_audit.py --limit 15
    python scripts/filter_mention_audit.py --from-json data/output/mention-audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_mentions as det  # noqa: E402
from src.agents.errors import AllKeysExhausted  # noqa: E402
from src.agents.gemini import GeminiAgent  # noqa: E402
from src.pipelines.morphology import fold  # noqa: E402
from src.state_manager import StateManager  # noqa: E402
from config.settings import get_settings  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Independent adjudicator prompt — not phase-1 extraction, not MentionsFill.
# Do not copy rules from src/pipelines/prompts.py into this block.
# ---------------------------------------------------------------------------
FILTER_SYSTEM = """\
You are a QA filter for an automated hadith-mention audit.

You are NOT extracting mentions. You are NOT rewriting mentions. You only
judge whether each *reported audit issue* is a real quality failure or a
false positive from a brittle heuristic.

For every issue, return verdict keep or drop:

  keep — the issue points at a genuine problem a human should review
  drop — the heuristic misfired; discard it from the human queue

False positives to DROP (do not send these to humans):
- Truncation flags caused by an ellipsis that also appears in the Arabic matn
  (editorial / Qur'anic abbreviation), when the translation is otherwise long
  and complete.
- “Uncovered root” / coverage flags about dialogue or grammar words, or about
  roots that are clearly already indexed by the mentions.
- “Too few mentions” on a short matn that already has one clear, well-evidenced
  topic.
- Any flag that punishes a *concept* mention whose label is not a literal matn
  substring, when its evidence span is present in the matn (or is a clear
  paraphrase / same verbal idea). Inferred concept labels with solid evidence
  are valid indexing — not errors.
- Salience / “not in matn” style complaints against such inferred concepts.

Real failures to KEEP:
- Contrast frames where the foil topic is indexed but the elevated side is not.
- Relative-clause / predicate hinges where mentions keep only the bare type and
  miss the conditioning claim.
- Framing-only tops (empty honorific compounds) with no real topic beside them.
- Mentions whose evidence is missing from the matn, or entity names invented
  (person/place/group must appear; concepts may be inferred).
- Genuinely truncated translations (stub endings, “continuation of…”, empty).

Set keep_row true only if at least one issue is keep.
Reasons must be one short sentence each. Do not invent new issues.
"""


class IssueVerdict(BaseModel):
    code: str
    verdict: Literal["keep", "drop"]
    reason: str = Field(default="", description="One short sentence")


class AuditFilterResult(BaseModel):
    keep_row: bool
    issues: list[IssueVerdict] = Field(default_factory=list)


def _matn_excerpt(matn: str, limit: int = 1800) -> str:
    folded = fold(matn)
    cut = folded.find(" قال ")
    if cut < 0:
        cut = folded.find(" يقول ")
    body = matn if cut < 0 else matn[max(0, cut - 20) :]
    body = body.strip()
    if len(body) <= limit:
        return body
    return body[:limit] + "\n…[matn truncated for filter]…"


def _user_prompt(row: dict, payload: dict) -> str:
    mentions = payload.get("mentions") or []
    mention_lines = []
    for m in mentions:
        if not isinstance(m, dict):
            continue
        mention_lines.append(
            f"- text={m.get('text')!r} type={m.get('type')!r} "
            f"salience={m.get('salience')!r} evidence={m.get('evidence')!r}"
        )
    issue_lines = []
    for issue in row.get("issues") or []:
        issue_lines.append(
            f"- code={issue.get('code')} detail={issue.get('detail')}"
        )
    return (
        f"marker: {row.get('marker')}\n"
        f"locator: {row.get('locator')}\n\n"
        f"MENTIONS:\n" + ("\n".join(mention_lines) or "(none)") + "\n\n"
        f"AUDIT ISSUES:\n" + ("\n".join(issue_lines) or "(none)") + "\n\n"
        f"ARABIC MATN (excerpt):\n{_matn_excerpt(str(payload.get('hadith') or ''))}\n"
    )


def _load_payload(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _run_noisy_audit(directory: Path, json_out: Path, top: int) -> list[dict]:
    code = det.main(
        [
            "--dir",
            str(directory),
            "--noisy",
            "--top",
            str(top),
            "--json-out",
            str(json_out),
            "--md-out",
            str(json_out.with_suffix(".md")),
        ]
    )
    if code != 0:
        raise RuntimeError(f"deterministic audit exited {code}")
    blob = json.loads(json_out.read_text(encoding="utf-8"))
    return list(blob.get("rows") or [])


def _filter_row(agent: GeminiAgent, row: dict, payload: dict) -> AuditFilterResult:
    return agent.complete_structured(
        _user_prompt(row, payload),
        AuditFilterResult,
        system=FILTER_SYSTEM,
    )


def _render_review(kept: list[dict], scanned_candidates: int) -> str:
    lines = [
        "# Mention audit — human review queue",
        "",
        f"LLM-filtered survivors from **{scanned_candidates}** noisy candidates "
        f"(showing **{len(kept)}**).",
        "",
        "Only issues the filter marked `keep` appear here. Inferred concepts "
        "with solid evidence should not appear solely for literal-label mismatch.",
        "",
    ]
    if not kept:
        lines.append("_Empty queue — nothing left to review._")
        lines.append("")
        return "\n".join(lines)

    for i, row in enumerate(kept, 1):
        lines.append(f"## {i}. risk={row.get('risk')} · {row.get('marker')} · {row.get('locator')}")
        lines.append("")
        lines.append(f"`{row.get('path')}`")
        lines.append("")
        lines.append(f"Mentions: {', '.join(row.get('mention_texts') or []) or '(none)'}")
        lines.append("")
        for issue in row.get("issues") or []:
            reason = issue.get("filter_reason") or ""
            suffix = f" — {reason}" if reason else ""
            lines.append(
                f"- **{issue.get('code')}** (+{issue.get('weight')}): "
                f"{issue.get('detail')}{suffix}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "data" / "output" / "phase1" / "hadith",
        help="Phase-1 hadith JSON directory",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Skip re-audit; load candidates from this noisy audit JSON",
    )
    parser.add_argument("--top", type=int, default=40, help="Max noisy candidates")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to send to the LLM")
    parser.add_argument(
        "--candidate-json",
        type=Path,
        default=ROOT / "data" / "output" / "mention-audit.json",
        help="Where to write/read noisy candidates",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=ROOT / "data" / "output" / "mention-audit-review.md",
        help="Human review markdown (survivors only)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "data" / "output" / "mention-audit-llm.json",
        help="Full keep/drop log",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build candidates only; do not call Gemini",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.from_json:
        blob = json.loads(args.from_json.read_text(encoding="utf-8"))
        candidates = list(blob.get("rows") or [])
    else:
        candidates = _run_noisy_audit(args.dir, args.candidate_json, args.top)

    if args.limit is not None:
        candidates = candidates[: max(0, args.limit)]

    print(f"candidates={len(candidates)}")
    if args.dry_run:
        print("dry-run: skipping LLM filter")
        return 0

    settings = get_settings()
    state = StateManager(settings.state_db)
    agent = GeminiAgent(state, settings)

    kept_rows: list[dict] = []
    log_rows: list[dict] = []
    dropped = 0

    try:
        for row in candidates:
            path = str(row.get("path") or "")
            payload = _load_payload(path)
            try:
                result = _filter_row(agent, row, payload)
            except Exception as exc:
                logger.warning("filter failed for %s: %s — keeping row for human", path, exc)
                kept = dict(row)
                for issue in kept.get("issues") or []:
                    issue["filter_verdict"] = "keep"
                    issue["filter_reason"] = f"filter error: {exc}"
                kept_rows.append(kept)
                log_rows.append({"path": path, "error": str(exc), "row": kept})
                continue

            verdict_by_code: dict[str, IssueVerdict] = {}
            for v in result.issues:
                verdict_by_code[v.code] = v

            surviving_issues = []
            drop_issues = []
            for issue in row.get("issues") or []:
                code = str(issue.get("code") or "")
                v = verdict_by_code.get(code)
                if v is None:
                    # Model omitted this code — keep conservatively.
                    issue = dict(issue)
                    issue["filter_verdict"] = "keep"
                    issue["filter_reason"] = "model omitted code; kept for safety"
                    surviving_issues.append(issue)
                    continue
                issue = dict(issue)
                issue["filter_verdict"] = v.verdict
                issue["filter_reason"] = v.reason
                if v.verdict == "keep":
                    surviving_issues.append(issue)
                else:
                    drop_issues.append(issue)

            keep_row = bool(surviving_issues) and (
                result.keep_row or bool(surviving_issues)
            )
            entry = {
                "path": path,
                "marker": row.get("marker"),
                "locator": row.get("locator"),
                "keep_row": keep_row,
                "kept_issues": surviving_issues,
                "dropped_issues": drop_issues,
                "model": result.model_dump(),
            }
            log_rows.append(entry)

            if keep_row and surviving_issues:
                out = dict(row)
                out["issues"] = surviving_issues
                out["risk"] = sum(int(i.get("weight") or 0) for i in surviving_issues)
                out["mention_texts"] = row.get("mention_texts") or []
                kept_rows.append(out)
            else:
                dropped += 1
    except AllKeysExhausted as exc:
        print(str(exc), file=sys.stderr)
        return 2

    kept_rows.sort(key=lambda r: (-int(r.get("risk") or 0), str(r.get("locator") or "")))

    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(
        _render_review(kept_rows, scanned_candidates=len(candidates)),
        encoding="utf-8",
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {
                "candidates": len(candidates),
                "kept": len(kept_rows),
                "dropped_rows": dropped,
                "rows": log_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"candidates={len(candidates)} kept={len(kept_rows)} "
        f"dropped_rows={dropped} written={args.md_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
