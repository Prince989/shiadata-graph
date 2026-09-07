"""Deterministic mention-quality audit over phase-1 hadith JSON.

Standalone: does not modify the ETL. Imports existing fold/ground helpers only.

Example:
    python scripts/audit_mentions.py
    python scripts/audit_mentions.py --dir data/output/phase1/hadith --top 40
    python scripts/audit_mentions.py --json-out data/output/mention-audit.json
"""

from __future__ import annotations

import argparse
import json
import re
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

# Isnad / deixis / grammar — never treated as missing topical coverage.
_STOP_ROOTS = frozenset(
    {
        "قول",
        "قال",  # morphology often leaves قال ≠ قول
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
    }
)

_ISSUE_WEIGHT = {
    "truncated_translation": 4,
    "too_few_mentions": 5,
    "ungrounded_mention": 5,
    "contrast_foil_without_elevated": 6,
    "framing_only_top": 4,
    "repeated_matn_root_uncovered": 3,
    "high_salience_not_in_matn": 5,
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
    # Skip isnad-ish prefix: before first قال / يقول often.
    folded = fold(matn)
    cut = folded.find(" قال ")
    if cut < 0:
        cut = folded.find(" يقول ")
    body = folded[cut + 1 :] if cut >= 0 else folded
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
        if n < 3:
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
        if len(issues) >= 4:
            break
    return issues


def _predicate_hinge_issues(matn: str, mentions: list[dict]) -> list[Issue]:
    """Flag when matn hinges on benefit-from-knowledge but mentions keep bare type.

    Detects ينتفع / انتفع (نفع stem in the surface form) near علم in the matn
    body; requires a mention that covers نفع / انتفاع / نافع, not only عالم via
    shared علم root. (morphology root(ينتفع) is unreliable → تفع.)
    """
    folded = fold(matn)
    cut = folded.find(" قال ")
    if cut < 0:
        cut = folded.find(" يقول ")
    body = folded[cut + 1 :] if cut >= 0 else folded
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
    # Framing-only if no other mention has salience >= 0.85
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


def audit_payload(payload: dict, path: Path) -> AuditRow | None:
    matn = str(payload.get("hadith") or "")
    if not matn.strip():
        return None
    mentions = [m for m in (payload.get("mentions") or []) if isinstance(m, dict)]
    issues: list[Issue] = []

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

    if len(mentions) < 2:
        issues.append(
            Issue(
                "too_few_mentions",
                f"only {len(mentions)} mention(s); expect ≥ 2",
                _ISSUE_WEIGHT["too_few_mentions"],
            )
        )

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
        sal = float(m.get("salience") or 0)
        text = str(m.get("text") or "")
        if sal >= 0.9 and text:
            content = [w for w in fold(text).split() if len(w) >= 3 and root(w) not in _STOP_ROOTS]
            matn_fold = fold(matn).replace(" ", "")
            if content and not any(fold(w).replace(" ", "") in matn_fold for w in content):
                issues.append(
                    Issue(
                        "high_salience_not_in_matn",
                        f"{text!r} salience={sal:.2f} has no content word in matn",
                        _ISSUE_WEIGHT["high_salience_not_in_matn"],
                    )
                )

    issues.extend(_contrast_issues(matn, mentions))
    issues.extend(_predicate_hinge_issues(matn, mentions))
    issues.extend(_coverage_issues(matn, mentions))
    issues.extend(_framing_issues(mentions))

    # Deduplicate by (code, detail)
    seen: set[tuple[str, str]] = set()
    unique: list[Issue] = []
    for issue in issues:
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


def render_markdown(rows: list[AuditRow], scanned: int, flagged: int) -> str:
    lines = [
        "# Mention audit",
        "",
        f"Scanned **{scanned}** hadith payloads; **{flagged}** with risk > 0 "
        f"(showing top **{len(rows)}**).",
        "",
        "Review this queue instead of reading every file. Codes:",
        "",
        "- `contrast_foil_without_elevated` — foil kept, elevated topic missing (mark-4 class)",
        "- `predicate_hinge_missing` — relative-clause hinge (e.g. ينتفع بعلمه) missing from mentions",
        "- `repeated_matn_root_uncovered` — recurring matn root with no mention",
        "- `framing_only_top` — top hit is كمال/فضل/… framing only",
        "- `truncated_translation` — FA/EN ellipsis / ادامه stubs",
        "- `too_few_mentions` / `ungrounded_mention` / `high_salience_not_in_matn`",
        "",
    ]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "data" / "output" / "phase1" / "hadith",
        help="Directory of phase-1 hadith JSON files",
    )
    parser.add_argument("--top", type=int, default=50, help="Max rows to print/write")
    parser.add_argument("--min-risk", type=int, default=1, help="Ignore rows below this risk")
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

    rows: list[AuditRow] = []
    scanned = 0
    for path, payload in iter_payloads(args.dir):
        scanned += 1
        row = audit_payload(payload, path)
        if row and row.risk >= args.min_risk:
            rows.append(row)

    rows.sort(key=lambda r: (-r.risk, r.locator, r.marker))
    top = rows[: max(0, args.top)]

    md = render_markdown(top, scanned, flagged=len(rows))
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(md, encoding="utf-8")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {"scanned": scanned, "flagged": len(rows), "rows": [r.to_public() for r in top]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"scanned={scanned} flagged={len(rows)} written={args.md_out}")
    if top:
        print("top risks:")
        for row in top[:15]:
            codes = ",".join(sorted({i.code for i in row.issues}))
            print(f"  {row.risk:>3}  {row.marker:<6} {row.locator}  [{codes}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
