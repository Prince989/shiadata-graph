"""Arabic stemming used as a stable hash, not as linguistics.

The goal is never a linguistically correct root. It is a key that two surface
forms of the same idea agree on. `الحساب` reduces to `حسب` here and
`حساب العباد` reduces to `حسب عبد`; whether a lexicographer would accept `حسب`
is irrelevant, because the only thing asked of it is that both sides land on the
same string. That lowers the bar from "needs a full morphological analyser" to
about a hundred lines with no dependencies.

Arabic is unusually well suited to this. Meaning lives in a consonantal root and
the patterns wrapped around it, so `العقل`, `عقول`, `العقول`, `عاقل`, `يعقل` and
the head of `عقل المرء` all collapse to one key -- which is exactly the
fragmentation that a list of approved labels was invented to paper over.
"""

from __future__ import annotations

import re
from functools import lru_cache

_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭـ]")
_NON_ARABIC = re.compile(r"[^ء-ي ]")

# Clitics and the definite article. Ordered longest-first so "وال" wins over "و".
_PREFIXES = ("وبال", "فبال", "وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل", "س")
_SUFFIXES = (
    "تموها", "كموها", "هما", "كما", "هم", "هن", "كم", "كن", "نا", "ها",
    "ات", "ون", "ين", "ان", "وا", "تم", "اء", "ة", "ه", "ي", "ا", "ن",
)

# Derivational patterns, written with ف/ع/ل marking the root slots. The value is
# the indices to keep. A word is tried against the patterns for its own length,
# in listed order, so resolution is deterministic even where a form is genuinely
# ambiguous -- determinism is what matters for a hash, not being right.
_PATTERNS: dict[int, tuple[tuple[str, tuple[int, ...]], ...]] = {
    4: (
        ("فعال", (0, 1, 3)),   # حساب  -> حسب
        ("فاعل", (0, 2, 3)),   # عاقل  -> عقل
        ("فعول", (0, 1, 3)),   # عقول  -> عقل
        ("فعيل", (0, 1, 3)),   # عليم  -> علم
        ("افعل", (1, 2, 3)),   # اكبر  -> كبر
        ("مفعل", (1, 2, 3)),   # مسجد  -> سجد
        ("تفعل", (1, 2, 3)),
        ("يفعل", (1, 2, 3)),
        ("فيعل", (0, 2, 3)),   # شيطن  -> شطن
        ("فعلن", (0, 1, 2)),
    ),
    5: (
        ("افتعل", (1, 3, 4)),  # اجتهد -> جهد
        ("انفعل", (2, 3, 4)),
        ("مفعول", (1, 2, 4)),  # معقول -> عقل
        ("تفعيل", (1, 2, 4)),
        ("مفعال", (1, 2, 4)),
        ("مفاعل", (1, 3, 4)),
        ("فعالة", (0, 1, 3)),
        ("افعال", (1, 2, 4)),
        ("تفاعل", (1, 3, 4)),
        ("فعلان", (0, 1, 2)),
        ("مفتعل", (1, 3, 4)),  # مجتهد -> جهد
    ),
    6: (
        ("افتعال", (1, 3, 5)),  # اجتهاد -> جهد
        ("انفعال", (2, 4, 5)),
        ("مفاعلة", (1, 3, 4)),
        ("استفعل", (3, 4, 5)),
        ("تفاعيل", (1, 3, 5)),
        ("مفعولة", (1, 2, 4)),
    ),
    7: (
        ("استفعال", (3, 4, 6)),
        ("مستفعل", (3, 4, 5)),
    ),
}

# Letters that pad a pattern rather than carry meaning. Used only by the
# fallback, when no pattern matched.
_WEAK = "اويتمنسهء"


def fold(text: str) -> str:
    """Strip vocalisation and unify letter shapes."""
    s = _DIACRITICS.sub("", text or "")
    s = re.sub(r"[آأإٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه")
    s = s.replace("ؤ", "و").replace("ئ", "ي")
    s = _NON_ARABIC.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _strip_affixes(word: str) -> str:
    for prefix in _PREFIXES:
        # A one-letter clitic needs four letters left behind, not three. With
        # only three the rule eats real radicals: كمال is فعال on the root كمل,
        # but ك is also the clitic "like", and stripping it yielded مال.
        floor = 4 if len(prefix) == 1 else 3
        if word.startswith(prefix) and len(word) - len(prefix) >= floor:
            word = word[len(prefix) :]
            break
    changed = True
    while changed and len(word) > 3:
        changed = False
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                changed = True
                break
    return word


def _match_pattern(word: str) -> str | None:
    for shape, keep in _PATTERNS.get(len(word), ()):
        ok = True
        for i, ch in enumerate(shape):
            if ch in "فعل":
                continue
            if word[i] != ch:
                ok = False
                break
        if ok:
            return "".join(word[i] for i in keep)
    return None


@lru_cache(maxsize=100_000)
def root(word: str) -> str:
    """A stable three-ish letter key for one Arabic word."""
    stem = _strip_affixes(fold(word))
    if len(stem) <= 3:
        return stem
    matched = _match_pattern(stem)
    if matched:
        return matched
    # Nothing matched: drop padding letters left to right until three remain.
    letters = list(stem)
    i = 0
    while len(letters) > 3 and i < len(letters):
        if letters[i] in _WEAK:
            letters.pop(i)
        else:
            i += 1
    return "".join(letters[:3])


def root_signature(label: str) -> tuple[str, ...]:
    """Root of every word in a label, in order. The identity key for a mention."""
    return tuple(root(word) for word in fold(label).split() if word)


def head_root(label: str) -> str:
    """Root of the first word. `عقل المرء` and `كمال العقل` both reach العقل by it."""
    signature = root_signature(label)
    return signature[0] if signature else ""


def is_tautology(label: str) -> bool:
    """True when a compound repeats one root, e.g. اجتهاد المجتهدين (ج-ه-د twice).

    These are always noise: a phrase that says the same thing twice describes
    this one sentence and can never be a shared index term.
    """
    signature = [r for r in root_signature(label) if r]
    return len(signature) > 1 and len(set(signature)) < len(signature)


def shares_root(left: str, right: str) -> bool:
    """True when two labels have any root in common."""
    return bool(set(root_signature(left)) & set(root_signature(right)) - {""})
