"""Show what the node gate does, without calling any API.

Run:  python check_nodes.py

Everything here is free and instant. It reads the real book off disk and runs
the real code -- no Gemini, no OpenAI, no cost.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.extractors.chunkers import strip_folklib_footnotes  # noqa: E402
from src.extractors.classification import attach_sections, page_headings  # noqa: E402
from src.extractors.quran_refs import page_quran_refs  # noqa: E402
from src.extractors.txt_parser import parse_txt  # noqa: E402
from src.core.candidates import Document, generate, summarise  # noqa: E402
from src.pipelines.grounding import ground_mentions  # noqa: E402
from src.pipelines.morphology import is_tautology, root  # noqa: E402
from src.pipelines.ontology import load_concept_catalog  # noqa: E402
from src.pipelines.resolver import Mention, resolve  # noqa: E402

KAFI1 = "data/raw_epubs/hadith/al-kafi-1.txt"


def line(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def test_1_vocabulary() -> None:
    line("1. THE AI NO LONGER PICKS FROM A LIST")
    print(f"\nThe curated catalog still exists ({len(load_concept_catalog())} entries),")
    print("but it is now an EXCEPTIONS table, not a gate. It holds only what")
    print("counting cannot discover on its own -- that قتل النفس and الانتحار are")
    print("one idea, for instance, which share no letters and no root.")
    print("\nEverything else is worked out from the corpus. Watch:")


def test_1b_roots() -> None:
    line("1b. ARABIC ROOTS DO MOST OF THE WORK, FOR FREE")
    families = [
        ["العقل", "عقول", "العقول", "عاقل", "يعقل", "المعقول"],
        ["الحساب", "يحاسب", "محاسبة"],
        ["اجتهاد", "المجتهدين", "مجتهد"],
    ]
    print("\nDifferent words, same idea -> same key:\n")
    for family in families:
        print(f"   {root(family[0]):<6} <-  {'  '.join(family)}")
    print("\nNobody had to list عقول as an alias of العقل. The language says it.")
    print("\nAnd a phrase repeating one root is always noise:")
    for label in ("اجتهاد المجتهدين", "خلق العقل"):
        verdict = "NOISE (same root twice)" if is_tautology(label) else "fine"
        print(f"   {label:<20} {verdict}")


def test_1c_resolver() -> None:
    line("1c. IDENTITY IS DECIDED BY THE WHOLE CORPUS, NOT PER HADITH")
    raw = [
        ("h1", "العقل", "concept"), ("h1", "كمال العقل", "concept"),
        ("h4", "عقل المرء", "concept"), ("h7", "قدر العقول", "concept"),
        ("h1", "خلق العقل", "concept"), ("h14", "خلق العقل", "concept"),
        ("h20", "خلق العقل", "concept"), ("h5", "عتاب الله", "concept"),
        ("a", "زرارة بن أعين", "person"), ("b", "زرارة", "person"),
    ]
    print("\nEach hadith said whatever its own wording suggested:\n")
    for doc, text, _ in raw:
        print(f"   {doc:<4} {text}")
    nodes = resolve([Mention(text=t, type=ty, doc_id=d) for d, t, ty in raw])
    print("\nAfter looking at all of them together:\n")
    for node in sorted(nodes.values(), key=lambda n: -n.df):
        tag = "  (a sub-topic)" if node.parent else ""
        print(f"   {node.label:<16} in {node.df} hadiths{tag}")
        print(f"        absorbed: {sorted(node.surfaces)}")
    print("\n   خلق العقل survived because THREE hadiths reached for it.")
    print("   عقل المرء did not, so it folded into العقل.")
    print("   عتاب الله was said once and built on nothing known -> dropped.")
    print("   زرارة and زرارة بن أعين became one man. Nobody listed him.")


def test_2_grounding() -> None:
    line("2. THE AI CANNOT MAKE THINGS UP")
    matn = (
        "أَحْمَدُ بْنُ إِدْرِيسَ عَنْ أَبِي عَبْدِ اللَّهِ ع قَالَ: مَا الْعَقْلُ "
        "قَالَ مَا عُبِدَ بِهِ الرَّحْمَنُ فَالَّذِي كَانَ فِي مُعَاوِيَةَ"
    )
    claimed = [
        {"text": "العقل", "type": "concept", "evidence": "ما عبد به الرحمن"},
        {"text": "معاوية", "type": "person", "evidence": "فالذي كان في معاوية"},
        {"text": "أبو عبد الله", "type": "person", "evidence": "عن أبي عبد الله"},
        {"text": "زرارة", "type": "person", "evidence": "حدثنا زرارة"},
        {"text": "الصبر", "type": "concept", "evidence": "و أمرهم بالصبر الجميل"},
    ]
    ravis = ["أَحْمَدُ بْنُ إِدْرِيسَ", "أَبُو عَبْدِ اللَّهِ (ع)"]
    print("\nThe hadith:\n")
    print(f"   {matn}")
    print("\nSuppose the AI claims all five of these:\n")
    for item in claimed:
        print(f"   {item['text']:<16} because: {item['evidence']}")
    kept, rejected = ground_mentions(claimed, matn, ravis)
    print("\nEvery claim is checked against the actual words of the hadith:\n")
    for item in kept:
        print(f"   KEPT     {item['text']}")
    for text, reason in rejected:
        print(f"   REJECTED {text:<16} ({reason})")
    print("\n   زرارة and الصبر were never in this hadith -- invented, so dropped.")
    print("   أبو عبد الله is real, but he is the one ANSWERING. A hadith is not")
    print("   about the man who narrated it, so he belongs in the chain, not the topics.")


def test_4_book_chapters() -> None:
    line("4. THE BOOK'S OWN CHAPTER TITLES (NO AI INVOLVED)")
    try:
        units = attach_sections(parse_txt(KAFI1))
    except FileNotFoundError:
        print(f"\n   {KAFI1} not found -- skipping.")
        return
    heads = [t for u in units for _, t in page_headings(u.text)]
    print(f"\nFound {len(heads)} chapter titles in volume 1. First five:\n")
    for title in heads[:5]:
        print(f"   {title}")
    print("\nEvery page inherits the title above it:\n")
    for unit in units[9:13]:
        print(f"   {unit.locator:<22} {unit.kitab}")
    print("\n   This is printed in the book. It is always right, and it is free.")
    print("   It means every hadith in a chapter is linked to every other one,")
    print("   even if the AI does a bad job on that page.")


def test_5_quran() -> None:
    line("5. QUR'AN VERSES FOUND IN THE TEXT")
    try:
        units = parse_txt(KAFI1)
    except FileNotFoundError:
        print(f"\n   {KAFI1} not found -- skipping.")
        return
    print()
    for unit in units[:14]:
        refs = page_quran_refs(unit.text)
        for marker, found in refs.items():
            print(f"   {unit.locator:<22} hadith {marker:<14} cites {found[:4]}")
    print("\n   Hadith 5 quotes a verse with no footnote at all; it was found by")
    print("   matching the words against the real Qur'an. The AI is not asked for")
    print("   verse numbers, so it cannot invent one.")


def test_6_footnote_leak() -> None:
    line("6. EDITOR FOOTNOTES STAY OUT OF THE HADITH")
    page = """11 - عِدَّةٌ مِنْ أَصْحَابِنَا رَفَعَهُ قَالَ قَالَ رَسُولُ اللَّهِ ص‌ مَا قَسَمَ اللَّهُ لِلْعِبَادِ شَيْئاً أَفْضَلَ مِنَ الْعَقْلِ.

[3] فهو يعلم ان الوسوسة من عمل الشيطان لما في قوله تعالى‌ « مِنْ

شَرِّ الْوَسْواسِ الْخَنَّاسِ الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ» و

لكنه لا يتمكن من طرده حين العمل.
"""
    clean = strip_folklib_footnotes(page)
    print("\nThe footnote quotes the Qur'an, which used to fool the cleaner.")
    print(f"\n   footnote text still present? {'الْخَنَّاسِ' in clean}")
    print(f"   editor's comment still present? {'طرده' in clean}")
    print(f"   the hadith itself survived?     {'مَا قَسَمَ اللَّهُ' in clean}")


def test_7_pairs() -> None:
    line("7. WHICH HADITHS GET COMPARED, AND WHY")
    docs = [
        Document(doc_id="A", nodes=["concept:الانتحار"], bab="باب من قتل نفسه",
                 ayahs=["4:29"], ravis=["زرارة"]),
        Document(doc_id="B", nodes=[], bab="باب من قتل نفسه", ayahs=["4:29"]),
        Document(doc_id="C", nodes=["concept:الانتحار"], bab="باب العذاب"),
        Document(doc_id="D", nodes=["concept:الصبر"], bab="باب الصبر"),
    ]
    print("\nFour hadiths:\n")
    for d in docs:
        topics = ", ".join(d.nodes) if d.nodes else "(none)"
        print(f"   {d.doc_id}  topics={topics:<20} chapter={d.bab:<18} verses={d.ayahs}")
    pairs = generate(docs)
    print("\nWorth comparing, best first:\n")
    for c in pairs:
        why = ", ".join(f"{k}" for k in c.reasons)
        print(f"   {c.left}-{c.right}   score {c.score:5.2f}   because: {why}")
    print("\n   A-B scores high even though B has NO topics at all: same chapter")
    print("   AND same verse. Under the old design a missing topic deleted the")
    print("   link entirely. Now it just ranks a little lower.")
    print("   D is never proposed -- nothing connects it to the others.")
    print("\n" + summarise(pairs, docs))


def main() -> None:
    for check in (
        test_1_vocabulary,
        test_1b_roots,
        test_1c_resolver,
        test_2_grounding,
        test_4_book_chapters,
        test_5_quran,
        test_6_footnote_leak,
        test_7_pairs,
    ):
        check()
    print("\n" + "=" * 68)
    print("Done. Nothing above called an API or cost anything.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
