"""Deterministic mention-quality audit over phase-1 hadith JSON.

Standalone: does not modify the ETL. Imports existing fold/ground helpers only.

Default mode keeps a short high-precision queue. Noisy heuristics
(coverage / broken salience substring / thin mention-count) stay off unless
you pass --noisy.

Example:
    python scripts/audit_mentions.py
    python scripts/audit_mentions.py --top 25
    python scripts/audit_mentions.py --noisy
    python scripts/audit_mentions.py --codes contrast_foil_without_elevated,predicate_hinge_missing
    python scripts/audit_mentions.py --json-out data/output/mention-audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.grounding import check_mention  # noqa: E402
from src.pipelines.llm_processor import translation_incomplete  # noqa: E402
from src.pipelines.morphology import fold, root, root_signature  # noqa: E402

# Obligation contrast frames (mark-4 class). Prefer these over أفضَل/خير —
# those produce many false positives on chained comparisons.
_CONTRAST_MARKERS = (
    "اوجب عليكم من",
    "اوجب من",
)

_FRAMING_HEADS = frozenset({"كمال", "صفه", "باب", "فضل", "صفت", "وجوب", "فرض"})

# Default review queue: high-precision claim/quality failures only.
# high_salience_not_in_matn is intentionally absent: inferred concepts with
# matn evidence (خشية الله ← لم يخف الله) are valid and must not pollute the queue.
_PRECISION_CODES = frozenset(
    {
        "contrast_foil_without_elevated",
        "predicate_hinge_missing",
        "framing_only_top",
        "ungrounded_mention",
        "truncated_translation",
    }
)

# Useful for debugging extractors; too many FPs for a human review queue.
# Still excludes high_salience_not_in_matn (literal-label check fights inference).
_NOISY_CODES = frozenset(
    {
        "repeated_matn_root_uncovered",
        "too_few_mentions",
    }
)

# Isnad / deixis / grammar — never treated as missing topical coverage.
_STOP_ROOTS = frozenset(
    {
        "قول",
        "قال",  # morphology often leaves قال ≠ قول
        "فقل",  # فقال → فقل
        "حدث",
        "خبر",
        "روي",
        "سمع",
        "الله",
        "الل",  # fold/root of الله
        "اله",
        "رسل",
        "نبي",
        "عبد",
        "حمد",
        "علي",
        "حسن",
        "حسين",
        "جعفر",
        "موسي",
        "رضا",
        "يحيي",
        "احمد",
        "محمد",
        "ابو",
        "ابي",
        "ابن",
        "بنى",
        "بنو",
        "هشم",
        "سلم",
        "حمز",
        "سحق",
        "سبع",
        "امر",  # أمير المؤمنين in isnad/address
        "موم",  # مؤمنين
        "قوم",
        "ناس",
        "رجل",
        "شئ",
        "شيء",
        "شيي",  # شيئا after fold
        "كان",
        "كون",
        "ليس",
        "هذا",
        "ذلك",
        "تلك",
        "الذي",
        "التى",
        "اللذ",
        "الذ",  # الذي → الذ
        "فيه",  # فيها / فيه deixis
        "ذكر",  # often discourse "ذكر" chains, not a topic by itself
        "عند",
        "الي",
        "علي",
        "بين",
        "لكم",
        "اهل",  # often اهله grammatical; topical اهل العلم usually compound
        "غير",
        "بعض",
        "كلل",
        "جميع",
        "ايي",
        "بما",
        "فاذ",  # فاذا
        "الا",
        "يقل",  # فيقول
        "تكن",
    }
)

_ISSUE_WEIGHT = {
    "truncated_translation": 4,
    "too_few_mentions": 3,
    "ungrounded_mention": 5,
    "contrast_foil_without_elevated": 6,
    "framing_only_top": 4,
    "repeated_matn_root_uncovered": 2,
    "high_salience_not_in_matn": 2,
    "predicate_hinge_missing": 6,
}


@dataclass
class Issue:
    code: str
    detail: str
    weight: int


@dataclass
class AuditRow:
    path: str
    marker: str
    locator: str
    risk: int
    issues: list[Issue] = field(default_factory=list)
    mention_texts: list[str] = field(default_factory=list)

    def to_public(self) -> dict:
        return {
            "path": self.path,
            "marker": self.marker,
            "locator": self.locator,
            "risk": self.risk,
            "mention_texts": self.mention_texts,
            "issues": [asdict(i) for i in self.issues],
        }


def _matn_body(matn: str) -> str:
    folded = fold(matn)
    cut = folded.find(" قال ")
    if cut < 0:
        cut = folded.find(" يقول ")
    return folded[cut + 1 :] if cut >= 0 else folded


def _mention_roots(mentions: list[dict]) -> set[str]:
    roots: set[str] = set()
    for m in mentions:
        text = str(m.get("text") or "")
        for r in root_signature(text):
            if r and len(r) >= 2:
                roots.add(r)
    return roots


def _covered(root_key: str, mention_roots: set[str], mentions: list[dict], matn_word: str) -> bool:
    if root_key in mention_roots:
        return True
    folded_word = fold(matn_word).replace(" ", "")
    for m in mentions:
        if root_key in set(root_signature(str(m.get("text") or ""))) - {""}:
            return True
        blob = fold(f"{m.get('text') or ''} {m.get('evidence') or ''}").replace(" ", "")
        if folded_word and folded_word in blob:
            return True
    return False


def _window_words(words: list[str], center: int, before: int, after: int) -> list[str]:
    start = max(0, center - before)
    end = min(len(words), center + after)
    return words[start:end]


def _contrast_issues(matn: str, mentions: list[dict]) -> list[Issue]:
    issues: list[Issue] = []
    folded = fold(matn)
    words = folded.split()
    mention_roots = _mention_roots(mentions)
    for marker in _CONTRAST_MARKERS:
        marker_words = marker.split()
        n = len(marker_words)
        for i in range(0, len(words) - n + 1):
            if words[i : i + n] != marker_words:
                continue
            left = _window_words(words, i, before=4, after=0)
            right = words[i + n : i + n + 3]
            left = [w for w in left if len(w) >= 3]
            right = [w for w in right if len(w) >= 3]
            if not left or not right:
                continue
            # Prefer the last content chunk on the left (closest to the marker).
            left_focus = left[-2:] if len(left) >= 2 else left
            left_roots = {root(w) for w in left_focus if root(w)} - {""} - _STOP_ROOTS
            right_roots = {root(w) for w in right if root(w)} - {""} - _STOP_ROOTS
            if not left_roots:
                continue
            # All elevated-side roots must be attested (union of mentions).
            # Intersection-only would let طلب المال "cover" طلب العلم via طلب.
            left_hit = left_roots <= mention_roots
            right_hit = bool(right_roots & mention_roots) if right_roots else False
            if right_hit and not left_hit:
                issues.append(
                    Issue(
                        code="contrast_foil_without_elevated",
                        detail=(
                            f"after «{marker}» foil roots {sorted(right_roots)} "
                            f"are mentioned but elevated roots {sorted(left_roots)} are not "
                            f"(left≈{' '.join(left_focus)}; right≈{' '.join(right)})"
                        ),
                        weight=_ISSUE_WEIGHT["contrast_foil_without_elevated"],
                    )
                )
    return issues


def _coverage_issues(matn: str, mentions: list[dict]) -> list[Issue]:
    """Noisy: dialogue morphology produces many false positives. Off by default."""
    body = _matn_body(matn)
    words = [w for w in body.split() if len(w) >= 3]
    counts: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for w in words:
        r = root(w)
        if not r or len(r) < 3 or r in _STOP_ROOTS:
            continue
        counts[r] += 1
        samples.setdefault(r, w)
    mention_roots = _mention_roots(mentions)
    issues: list[Issue] = []
    for r, n in counts.most_common():
        if n < 5:  # stricter than the old ≥3 threshold
            break
        sample = samples[r]
        if _covered(r, mention_roots, mentions, sample):
            continue
        issues.append(
            Issue(
                code="repeated_matn_root_uncovered",
                detail=f"root {r} appears x{n} in matn body (e.g. {sample}) but no mention covers it",
                weight=_ISSUE_WEIGHT["repeated_matn_root_uncovered"],
            )
        )
        if len(issues) >= 3:
            break
    return issues


def _predicate_hinge_issues(matn: str, mentions: list[dict]) -> list[Issue]:
    """Flag when matn hinges on benefit-from-knowledge but mentions keep bare type.

    Detects ينتفع / انتفع (نفع stem in the surface form) near علم in the matn
    body; requires a mention that covers نفع / انتفاع / نافع, not only عالم via
    shared علم root. (morphology root(ينتفع) is unreliable → تفع.)
    """
    body = _matn_body(matn)
    words = body.split()

    def _naf_stem(w: str) -> bool:
        f = fold(w)
        # ينتفع = ي+نتفع (ت between ن and ف) — contiguous "نفع" is absent.
        return "نتفع" in f or "نفع" in f or root(w) == "نفع"

    hinge = False
    for i, w in enumerate(words):
        if not _naf_stem(w):
            continue
        window = words[max(0, i - 3) : i + 4]
        if any(root(x) == "علم" or "علم" in fold(x) for x in window):
            hinge = True
            break
    if not hinge:
        return []

    mention_roots = _mention_roots(mentions)
    joined = " ".join(fold(str(m.get("text") or "")) for m in mentions)
    if (
        "نفع" in mention_roots
        or "انتفاع" in joined
        or "نافع" in joined
        or "نفع" in joined
        or "نتفع" in joined  # ينتفع / انتفع inside a compound mention
    ):
        return []

    typed = False
    for m in mentions:
        if float(m.get("salience") or 0) < 0.8:
            continue
        text = fold(str(m.get("text") or ""))
        roots = set(root_signature(text)) - {""}
        if "علم" in roots and "نفع" not in roots and "نفع" not in text:
            typed = True
            break
    if not typed:
        return []

    return [
        Issue(
            code="predicate_hinge_missing",
            detail=(
                "matn has ينتفع/نفع near علم, but mentions keep a high-salience "
                "علم-type noun without انتفاع/نفع/نافع"
            ),
            weight=_ISSUE_WEIGHT["predicate_hinge_missing"],
        )
    ]


def _framing_issues(mentions: list[dict]) -> list[Issue]:
    if not mentions:
        return []
    top = max(mentions, key=lambda m: float(m.get("salience") or 0))
    sal = float(top.get("salience") or 0)
    if sal < 0.85:
        return []
    text = str(top.get("text") or "")
    words = fold(text).split()
    if len(words) < 2:
        return []
    head = words[0]
    if head not in _FRAMING_HEADS:
        return []
    others = [
        m
        for m in mentions
        if m is not top and float(m.get("salience") or 0) >= 0.85
    ]
    if others:
        return []
    return [
        Issue(
            code="framing_only_top",
            detail=f"top mention is framing compound {text!r} @ {sal:.2f} with no other high-salience topic",
            weight=_ISSUE_WEIGHT["framing_only_top"],
        )
    ]


def _too_few_issues(matn: str, mentions: list[dict]) -> list[Issue]:
    """Noisy by default. Empty mentions always; single mention only on long matn."""
    n = len(mentions)
    if n == 0:
        return [
            Issue(
                "too_few_mentions",
                "no mentions",
                _ISSUE_WEIGHT["too_few_mentions"],
            )
        ]
    if n >= 2:
        return []
    body_words = [w for w in _matn_body(matn).split() if len(w) >= 3]
    if len(body_words) < 60:
        return []
    return [
        Issue(
            "too_few_mentions",
            f"only 1 mention on a long matn ({len(body_words)} body words)",
            _ISSUE_WEIGHT["too_few_mentions"],
        )
    ]


def _salience_ground_issues(matn: str, mentions: list[dict]) -> list[Issue]:
    """Noisy: require content-root overlap with matn (not broken substrings)."""
    issues: list[Issue] = []
    matn_roots = {
        root(w)
        for w in fold(matn).split()
        if len(w) >= 3 and root(w) and root(w) not in _STOP_ROOTS
    }
    for m in mentions:
        sal = float(m.get("salience") or 0)
        text = str(m.get("text") or "")
        if sal < 0.9 or not text:
            continue
        content_roots = {
            root(w)
            for w in fold(text).split()
            if len(w) >= 3 and root(w) and root(w) not in _STOP_ROOTS
        }
        if not content_roots:
            continue
        if content_roots & matn_roots:
            continue
        issues.append(
            Issue(
                "high_salience_not_in_matn",
                f"{text!r} salience={sal:.2f} has no content root in matn",
                _ISSUE_WEIGHT["high_salience_not_in_matn"],
            )
        )
    return issues


def audit_payload(
    payload: dict,
    path: Path,
    *,
    enabled_codes: frozenset[str] | None = None,
) -> AuditRow | None:
    matn = str(payload.get("hadith") or "")
    if not matn.strip():
        return None
    mentions = [m for m in (payload.get("mentions") or []) if isinstance(m, dict)]
    codes = enabled_codes if enabled_codes is not None else _PRECISION_CODES

    issues: list[Issue] = []

    if "truncated_translation" in codes:
        if translation_incomplete(str(payload.get("hadith_fa") or "")):
            issues.append(
                Issue(
                    "truncated_translation",
                    "hadith_fa looks truncated/summarized",
                    _ISSUE_WEIGHT["truncated_translation"],
                )
            )
        if translation_incomplete(str(payload.get("hadith_en") or "")):
            issues.append(
                Issue(
                    "truncated_translation",
                    "hadith_en looks truncated/summarized",
                    _ISSUE_WEIGHT["truncated_translation"],
                )
            )

    if "too_few_mentions" in codes:
        issues.extend(_too_few_issues(matn, mentions))

    if "ungrounded_mention" in codes:
        for m in mentions:
            reason = check_mention(m, matn)
            if reason:
                issues.append(
                    Issue(
                        "ungrounded_mention",
                        f"{m.get('text')!r}: {reason}",
                        _ISSUE_WEIGHT["ungrounded_mention"],
                    )
                )

    if "high_salience_not_in_matn" in codes:
        issues.extend(_salience_ground_issues(matn, mentions))

    if "contrast_foil_without_elevated" in codes:
        issues.extend(_contrast_issues(matn, mentions))
    if "predicate_hinge_missing" in codes:
        issues.extend(_predicate_hinge_issues(matn, mentions))
    if "repeated_matn_root_uncovered" in codes:
        issues.extend(_coverage_issues(matn, mentions))
    if "framing_only_top" in codes:
        issues.extend(_framing_issues(mentions))

    seen: set[tuple[str, str]] = set()
    unique: list[Issue] = []
    for issue in issues:
        if issue.code not in codes:
            continue
        key = (issue.code, issue.detail)
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)

    risk = sum(i.weight for i in unique)
    if risk <= 0:
        return None
    return AuditRow(
        path=str(path),
        marker=str(payload.get("marker") or ""),
        locator=str(payload.get("locator") or ""),
        risk=risk,
        issues=unique,
        mention_texts=[str(m.get("text") or "") for m in mentions],
    )


def iter_payloads(directory: Path):
    for path in sorted(directory.rglob("*.json")):
        if path.name in {"nodes.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if "hadith" not in data and "mentions" not in data:
            continue
        yield path, data


def render_markdown(
    rows: list[AuditRow],
    scanned: int,
    flagged: int,
    *,
    mode: str,
) -> str:
    lines = [
        "# Mention audit",
        "",
        f"Mode: **{mode}**. Scanned **{scanned}** hadith payloads; "
        f"**{flagged}** with risk > 0 (showing top **{len(rows)}**).",
        "",
        "Default mode is a short high-precision queue. Pass `--noisy` for "
        "coverage / thin-count heuristics (literal-label salience stays off). "
        "For a human review queue run `python scripts/filter_mention_audit.py`.",
        "",
        "Codes in this report:",
        "",
    ]
    shown = sorted({i.code for row in rows for i in row.issues})
    if not shown:
        lines.append("_No issues under the active code filter._")
        lines.append("")
    else:
        for code in shown:
            lines.append(f"- `{code}`")
        lines.append("")

    for i, row in enumerate(rows, 1):
        lines.append(f"## {i}. risk={row.risk} · {row.marker} · {row.locator}")
        lines.append("")
        lines.append(f"`{row.path}`")
        lines.append("")
        lines.append(f"Mentions: {', '.join(row.mention_texts) or '(none)'}")
        lines.append("")
        for issue in row.issues:
            lines.append(f"- **{issue.code}** (+{issue.weight}): {issue.detail}")
        lines.append("")
    return "\n".join(lines)


def _resolve_codes(*, noisy: bool, codes: str | None) -> tuple[frozenset[str], str]:
    if codes:
        selected = frozenset(c.strip() for c in codes.split(",") if c.strip())
        return selected, f"custom ({', '.join(sorted(selected))})"
    if noisy:
        return _PRECISION_CODES | _NOISY_CODES, "noisy (precision + coverage/count)"
    return _PRECISION_CODES, "precision (default)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "data" / "output" / "phase1" / "hadith",
        help="Directory of phase-1 hadith JSON files",
    )
    parser.add_argument("--top", type=int, default=25, help="Max rows to print/write")
    parser.add_argument("--min-risk", type=int, default=1, help="Ignore rows below this risk")
    parser.add_argument(
        "--noisy",
        action="store_true",
        help="Also enable coverage / salience / too-few heuristics",
    )
    parser.add_argument(
        "--codes",
        type=str,
        default=None,
        help="Comma-separated issue codes to enable (overrides --noisy)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=ROOT / "data" / "output" / "mention-audit.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    args = parser.parse_args(argv)

    if not args.dir.exists():
        print(f"directory not found: {args.dir}", file=sys.stderr)
        return 1

    enabled, mode = _resolve_codes(noisy=bool(args.noisy), codes=args.codes)

    rows: list[AuditRow] = []
    scanned = 0
    for path, payload in iter_payloads(args.dir):
        scanned += 1
        row = audit_payload(payload, path, enabled_codes=enabled)
        if row and row.risk >= args.min_risk:
            rows.append(row)

    rows.sort(key=lambda r: (-r.risk, r.locator, r.marker))
    top = rows[: max(0, args.top)]

    md = render_markdown(top, scanned, flagged=len(rows), mode=mode)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(md, encoding="utf-8")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "scanned": scanned,
                    "flagged": len(rows),
                    "mode": mode,
                    "codes": sorted(enabled),
                    "rows": [r.to_public() for r in top],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"scanned={scanned} flagged={len(rows)} mode={mode} written={args.md_out}")
    if top:
        print("top risks:")
        for row in top[:15]:
            codes = ",".join(sorted({i.code for i in row.issues}))
            print(f"  {row.risk:>3}  {row.marker:<6} {row.locator}  [{codes}]")
    else:
        print("no rows under active filters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
