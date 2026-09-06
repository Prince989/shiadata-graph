"""Morphology, corpus-wide resolution, and candidate generation.

Every label in here is verbatim from a real extraction run.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.candidates import Document, generate, idf_table, summarise
from src.extractors.quran_refs import match_quran, match_quran_roots
from src.pipelines.morphology import is_tautology, root, root_signature
from src.pipelines.resolve_pass import collect_mentions
from src.pipelines.resolver import Mention, resolve


# --------------------------------------------------------------- morphology
def test_derivations_of_one_root_collapse_to_one_key():
    """The fragmentation that a curated word list existed to paper over."""
    for family in (
        ["العقل", "عقول", "العقول", "عاقل", "يعقل", "المعقول", "بالعقل", "للعقل"],
        ["الحساب", "يحاسب", "محاسبة"],
        ["الصبر", "الصابرين", "صابر"],
        ["اجتهاد", "المجتهدين", "مجتهد"],
    ):
        keys = {root(word) for word in family}
        assert len(keys) == 1, f"{family} split into {keys}"


def test_clitic_stripping_does_not_eat_a_radical():
    # ك is both a radical here and the clitic "like"; كمال is فعال on كمل.
    assert root("كمال") == root("يكمل")
    assert root("كمال") != root("مال")


def test_tautology_is_detected_without_a_word_list():
    # اجتهاد المجتهدين is ج-ه-د twice: it describes one sentence, never a topic.
    assert is_tautology("اجتهاد المجتهدين")
    assert not is_tautology("خلق العقل")
    assert not is_tautology("محبة أهل البيت")


def test_root_signature_keeps_word_order():
    assert root_signature("خلق العقل") == ("خلق", "عقل")


# ----------------------------------------------------------------- resolver
def _m(doc: str, text: str, node_type: str = "concept") -> Mention:
    return Mention(text=text, type=node_type, doc_id=doc)


def test_phrasing_folds_into_the_topic_it_is_built_on():
    """عقل المرء، كمال العقل، قدر العقول are one sentence's grammar, not topics."""
    nodes = resolve(
        [
            _m("h1", "العقل"),
            _m("h1", "كمال العقل"),
            _m("h4", "عقل المرء"),
            _m("h7", "قدر العقول"),
            _m("h11", "العقول"),
        ]
    )
    assert len(nodes) == 1
    node = next(iter(nodes.values()))
    assert node.label == "العقل"
    assert node.df == 4
    assert {"عقل المرء", "كمال العقل", "قدر العقول"} <= node.surfaces


def test_a_compound_that_recurs_keeps_its_own_identity():
    """خلق العقل and عقل المرء are the same shape; only the corpus separates them."""
    nodes = resolve(
        [
            _m("h1", "خلق العقل"),
            _m("h14", "خلق العقل"),
            _m("h20", "خلق العقل"),
            _m("h1", "عقل المرء"),
            _m("h2", "العقل"),
        ]
    )
    labels = {n.label: n for n in nodes.values()}
    assert "خلق العقل" in labels
    assert labels["خلق العقل"].df == 3
    # It hangs under العقل rather than competing with it.
    assert labels["خلق العقل"].parent is not None
    # And the one-off phrasing did not survive.
    assert "عقل المرء" not in labels


def test_a_singleton_anchored_to_nothing_is_dropped():
    """عتاب الله: said once, built on no known topic, reachable by nobody."""
    nodes = resolve([_m("h5", "عتاب الله"), _m("h5", "العزيمة")])
    assert {n.label for n in nodes.values()} == {"العزيمة"}


def test_a_curated_parent_is_created_even_if_never_seen_alone():
    """حساب العباد must still reach الحساب, which the catalog knows."""
    nodes = resolve([_m("h7", "حساب العباد")])
    assert {n.label for n in nodes.values()} == {"الحساب"}


def test_people_resolve_without_being_enumerated_anywhere():
    """The معاوية problem: no gazetteer can hold every name in these books."""
    nodes = resolve(
        [
            _m("a", "زرارة بن أعين", "person"),
            _m("b", "زرارة", "person"),
            _m("c", "زرارة بن أعين", "person"),
            _m("d", "أبو ذر الغفاري", "person"),
            _m("e", "أبو ذر", "person"),
        ]
    )
    labels = {n.label: n for n in nodes.values()}
    assert labels["زرارة بن أعين"].df == 3
    assert labels["أبو ذر الغفاري"].df == 2
    assert "زرارة" not in labels and "أبو ذر" not in labels


def test_generic_head_does_not_split_an_event():
    nodes = resolve([_m("h", "واقعة صفين", "event"), _m("i", "صفين", "event")])
    assert len(nodes) == 1
    assert next(iter(nodes.values())).df == 2


def test_curated_aliases_still_do_what_clustering_cannot():
    """قتل النفس and الانتحار share no root and no wording."""
    nodes = resolve([_m("a", "قتل النفس"), _m("b", "الانتحار")])
    assert len(nodes) == 1
    node = next(iter(nodes.values()))
    assert node.label == "الانتحار" and node.df == 2 and node.curated


def test_tautologies_never_become_nodes():
    nodes = resolve([_m("h11", "اجتهاد المجتهدين"), _m("h11", "العقل")])
    assert {n.label for n in nodes.values()} == {"العقل"}


# ---------------------------------------------------------------- resolve pass
def test_legacy_semantic_nodes_are_upcast_to_mentions(tmp_path: Path):
    """A corpus part-extracted under the old contract still resolves."""
    book = tmp_path / "hadith"
    book.mkdir()
    (book / "a.json").write_text(
        json.dumps(
            {
                "marker": "1-",
                "locator": "p10",
                "semantic_nodes": [
                    {"node": "العقل", "type": "concept", "role": "primary"},
                    {"node": "معاوية", "type": "person", "role": "secondary"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    mentions, sources = collect_mentions(tmp_path)
    assert len(sources) == 1
    by_text = {m.text: m for m in mentions}
    # primary/secondary maps onto salience so old and new payloads compare.
    assert by_text["العقل"].salience > by_text["معاوية"].salience


# --------------------------------------------------------------- candidates
def test_idf_makes_a_rare_shared_node_outweigh_a_common_one():
    docs = [Document(doc_id=f"d{i}", nodes=["العقل"]) for i in range(20)]
    docs[0].nodes.append("الانتحار")
    docs[1].nodes.append("الانتحار")
    idf = idf_table(docs)
    assert idf["node::الانتحار"] > idf["node::العقل"]


def test_a_missed_node_does_not_delete_the_relation():
    """The point of scoring on several signals at once.

    d1 and d2 are the same narration pair with one node missing on d2. It ranks
    lower, but it is still proposed -- under node-only bucketing it vanished.
    """
    docs = [
        Document(doc_id="d1", nodes=["الانتحار"], bab="باب الانتحار", ayahs=["4:29"]),
        Document(doc_id="d2", nodes=[], bab="باب الانتحار", ayahs=["4:29"]),
        Document(doc_id="d3", nodes=["الصبر"], bab="باب الصبر"),
    ]
    pairs = {(c.left, c.right): c for c in generate(docs)}
    assert ("d1", "d2") in pairs
    assert set(pairs[("d1", "d2")].reasons) >= {"ayah", "bab"}
    assert ("d1", "d3") not in pairs


def test_kitab_never_blocks_even_though_it_scores():
    """A kitab of 1,607 narrations is 1.3M pairs from one key."""
    docs = [Document(doc_id=f"d{i}", kitab="كتاب العقل") for i in range(50)]
    assert generate(docs) == []
    docs[0].bab = docs[1].bab = "باب خلق العقل"
    pairs = generate(docs)
    assert [(c.left, c.right) for c in pairs] == [("d0", "d1")]
    assert "kitab" in pairs[0].reasons


def test_summary_reports_the_budget():
    docs = [
        Document(doc_id="d1", nodes=["الانتحار"]),
        Document(doc_id="d2", nodes=["الانتحار"]),
    ]
    text = summarise(generate(docs), docs)
    assert "candidates" in text and "all-pairs" in text


# -------------------------------------------------------------- integration
def _resolved_chunk(cid: str, nodes: list[dict], **payload_extra):
    from src.state_manager import ChunkRecord, ChunkStatus

    payload = {"hadith": "متن", "nodes": nodes, **payload_extra}
    return ChunkRecord(
        id=cid,
        book_id="hadith",
        pipeline="hadith",
        locator="p",
        source_path="x.txt",
        text="متن طويل بما يكفي",
        status=ChunkStatus.EMBEDDED,
        payload_json=json.dumps(payload, ensure_ascii=False),
    )


def test_resolved_nodes_win_over_the_legacy_gate():
    """Two identity systems must not both feed the graph.

    A payload carrying resolver output must never be re-run through the
    per-hadith ontology gate, or the export mixes canonical keys with
    alias-table labels.
    """
    from src.core.vector_engine import bucket_keys_for_chunk, graph_nodes_for_chunk

    chunk = _resolved_chunk(
        "h1",
        [{"key": "concept:@عقل", "label": "العقل", "type": "concept", "weight": 0.9}],
        semantic_nodes=[{"node": "الصبر", "type": "concept", "role": "primary"}],
    )
    nodes = graph_nodes_for_chunk(chunk)
    assert [n["node"] for n in nodes] == ["العقل"]
    assert "الصبر" not in str(nodes)
    # Buckets key on the canonical resolver key, not the display label.
    assert "concept:@عقل" in bucket_keys_for_chunk(chunk)


def test_legacy_payloads_still_export_when_the_resolver_has_not_run():
    from src.core.vector_engine import graph_nodes_for_chunk

    chunk = _resolved_chunk(
        "h2",
        [],
        semantic_nodes=[{"node": "الصبر", "type": "concept", "role": "primary"}],
        ravis=[],
    )
    assert [n["node"] for n in graph_nodes_for_chunk(chunk)] == ["الصبر"]


def test_phase2_projects_chunks_into_linker_documents():
    from src.core.edge_classifier import documents_for

    chunk = _resolved_chunk(
        "h3",
        [{"key": "concept:@عقل", "label": "العقل", "type": "concept", "weight": 0.9}],
        quran_refs=["2:269"],
        ravis=["زرارة"],
        kitab="كتاب العقل",
        bab="باب خلق العقل",
    )
    doc = documents_for([chunk])[0]
    assert doc.nodes == ["concept:@عقل"]
    assert doc.ayahs == ["2:269"]
    assert doc.ravis == ["زرارة"]
    assert doc.bab == "باب خلق العقل" and doc.kitab == "كتاب العقل"


def test_resolver_output_is_not_reprocessed_by_the_old_remap():
    from src.core.phase2 import remap_existing_semantic_nodes
    from src.state_manager import StateManager

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        state = StateManager(Path(tmp) / "state.db")
        try:
            state.upsert_chunks(
                [
                    {
                        "id": "h9",
                        "book_id": "hadith",
                        "pipeline": "hadith",
                        "locator": "p",
                        "source_path": "x.txt",
                        "text": "متن",
                    }
                ]
            )
            resolved = {
                "hadith": "متن",
                "nodes": [{"key": "concept:@عقل", "label": "العقل", "type": "concept"}],
                "semantic_nodes": [{"node": "فضيلة العقل", "type": "concept"}],
            }
            from src.state_manager import ChunkStatus

            state.mark("h9", ChunkStatus.PROCESSED_PHASE1, payload=resolved)
            assert remap_existing_semantic_nodes(state, Path(tmp)) == 0
            after = state.get_chunk("h9").payload()
            assert after["nodes"][0]["key"] == "concept:@عقل"
        finally:
            state.close()


# ---------------------------------------------------------------- grounding
def test_invented_mentions_are_dropped():
    from src.pipelines.grounding import ground_mentions

    matn = "قُلْتُ لَهُ مَا الْعَقْلُ قَالَ مَا عُبِدَ بِهِ الرَّحْمَنُ فَالَّذِي كَانَ فِي مُعَاوِيَةَ"
    kept, rejected = ground_mentions(
        [
            {"text": "العقل", "type": "concept", "evidence": "ما عبد به الرحمن"},
            {"text": "معاوية", "type": "person", "evidence": "فالذي كان في معاوية"},
            {"text": "زرارة", "type": "person", "evidence": "حدثنا زرارة"},
            {"text": "الصبر", "type": "concept", "evidence": "و أمرهم بالصبر الجميل"},
        ],
        matn,
    )
    assert [m["text"] for m in kept] == ["العقل", "معاوية"]
    assert {text for text, _ in rejected} == {"زرارة", "الصبر"}


def test_narrators_are_still_filtered_under_the_mention_contract():
    """The speaker filter used to live in the gate the mention path bypasses."""
    from src.pipelines.grounding import ground_mentions

    matn = "أَحْمَدُ بْنُ إِدْرِيسَ عَنْ أَبِي عَبْدِ اللَّهِ ع قَالَ مَا الْعَقْلُ"
    kept, rejected = ground_mentions(
        [
            {"text": "العقل", "type": "concept", "evidence": "ما العقل"},
            {"text": "أبو عبد الله", "type": "person", "evidence": "عن أبي عبد الله"},
        ],
        matn,
        ["أَحْمَدُ بْنُ إِدْرِيسَ", "أَبُو عَبْدِ اللَّهِ (ع)"],
    )
    assert [m["text"] for m in kept] == ["العقل"]
    assert rejected == [("أبو عبد الله", "narrator, not a subject")]


# ------------------------------------------------- the mention contract runs
def test_mentions_and_quotes_survive_the_phase1_flush():
    """The accumulator used to copy only semantic_nodes, so mentions died here.

    Everything downstream -- resolve-nodes, the node table, the linker -- reads
    what `assemble()` emits, so a mention dropped at the flush never existed.
    """
    from src.pipelines.hadith_accumulator import consume_page
    from src.pipelines.llm_processor import needs_enrichment

    page = (
        "3 - أَحْمَدُ بْنُ إِدْرِيسَ عَنْ أَبِي عَبْدِ اللَّهِ ع قَالَ: مَا الْعَقْلُ "
        "قَالَ مَا عُبِدَ بِهِ الرَّحْمَنُ فَالَّذِي كَانَ فِي مُعَاوِيَةَ فَقَالَ تِلْكَ النَّكْرَاءُ.\n"
    )
    items = [
        {
            "marker": "3 -",
            "hadith": "x",
            "hadith_fa": "ف",
            "hadith_en": "e",
            "ravis": ["أَحْمَدُ بْنُ إِدْرِيسَ", "أَبُو عَبْدِ اللَّهِ (ع)"],
            "mentions": [
                {"text": "العقل", "type": "concept", "salience": 0.9,
                 "evidence": "ما عبد به الرحمن"},
                {"text": "معاوية", "type": "person", "salience": 0.4,
                 "evidence": "فالذي كان في معاوية"},
            ],
            "quotes": [{"text": "فاعتبروا يا أولي الأبصار", "kind": "quran"}],
        }
    ]
    flushed, buf = consume_page(
        "جلد 1 - صفحه 11", page, items, None, None,
        {"3 -": ["59:2"]}, ("كتاب العقل", "باب صفة العقل"),
    )
    assert buf is None
    hadith = flushed[0]
    assert [m["text"] for m in hadith["mentions"]] == ["العقل", "معاوية"]
    assert [q["text"] for q in hadith["quotes"]] == ["فاعتبروا يا أولي الأبصار"]
    assert hadith["quran_refs"] == ["59:2"]
    assert hadith["kitab"] == "كتاب العقل"
    # And a payload carrying mentions must not look hollow, or every hadith
    # extracted under the new contract pays for a second model call.
    assert needs_enrichment(hadith) is False


def test_mentions_union_across_a_spanning_hadith():
    """Each page fragment sees only its own half of the matn."""
    from src.pipelines.hadith_accumulator import OpenHadith

    buf = OpenHadith(marker="8-", page_start="p11", page_end="p11")
    buf.append_slice("p11", "أَوَّلُ الْمَتْنِ", mentions=[{"text": "العقل", "type": "concept"}])
    buf.append_slice("p12", "آخِرُ الْمَتْنِ", mentions=[{"text": "الثواب", "type": "concept"}])
    assert [m["text"] for m in buf.mentions_seed] == ["العقل", "الثواب"]


def test_buffer_round_trip_keeps_mentions():
    """A run resumed mid-hadith must not lose the first page's mentions."""
    from src.pipelines.hadith_accumulator import OpenHadith

    buf = OpenHadith(marker="8-", page_start="p11", page_end="p11")
    buf.append_slice(
        "p11", "متن", mentions=[{"text": "العقل", "type": "concept"}],
        quotes=[{"text": "قول", "kind": "quran"}],
    )
    restored = OpenHadith.from_dict(json.loads(json.dumps(buf.to_dict(), ensure_ascii=False)))
    assert [m["text"] for m in restored.mentions_seed] == ["العقل"]
    assert [q["text"] for q in restored.quotes_seed] == ["قول"]


def test_mistyped_mentions_resolve_to_one_identity():
    """place:الجنة and concept:الجنة were parallel, unlinkable identities."""
    nodes = resolve(
        [
            _m("a", "الجنة", "place"),
            _m("b", "الجنة", "concept"),
            _m("c", "يوم القيامة", "event"),
            _m("d", "القيامة", "concept"),
            _m("e", "الشيطان", "concept"),
            _m("f", "الشيطان", "person"),
        ]
    )
    by_label = {n.label: n for n in nodes.values()}
    assert by_label["الجنة"].type == "concept" and by_label["الجنة"].df == 2
    assert by_label["القيامة"].type == "concept" and by_label["القيامة"].df == 2
    # And the reverse direction: a named being emitted as an idea.
    assert by_label["الشيطان"].type == "person"


def test_narrator_filter_resolves_the_mention_through_the_gazetteer():
    """A raw string compare lets the speaker back in under another of his names."""
    from src.pipelines.grounding import ground_mentions

    matn = "أَحْمَدُ بْنُ إِدْرِيسَ عَنْ أَبِي عَبْدِ اللَّهِ ع قَالَ جَعْفَرُ بْنُ مُحَمَّدٍ"
    kept, rejected = ground_mentions(
        [
            {"text": "العقل", "type": "concept", "evidence": "عن أبي عبد الله"},
            {"text": "جعفر بن محمد", "type": "person", "evidence": "قال جعفر بن محمد"},
        ],
        matn,
        ["أَحْمَدُ بْنُ إِدْرِيسَ", "أَبُو عَبْدِ اللَّهِ (ع)"],
    )
    assert [m["text"] for m in kept] == ["العقل"]
    assert rejected == [("جعفر بن محمد", "narrator, not a subject")]


def test_missing_evidence_no_longer_skips_the_check():
    from src.pipelines.grounding import ground_mentions

    matn = "قُلْتُ لَهُ مَا الْعَقْلُ قَالَ مَا عُبِدَ بِهِ الرَّحْمَنُ"
    kept, rejected = ground_mentions(
        [
            {"text": "العقل", "type": "concept", "evidence": ""},
            {"text": "الصبر", "type": "concept", "evidence": ""},
        ],
        matn,
    )
    assert [m["text"] for m in kept] == ["العقل"]
    assert rejected == [("الصبر", "no evidence and term not in matn")]


def test_hierarchy_reaches_the_linker():
    """parent and broader were stored on the node table and used by nothing."""
    from src.core.vector_engine import bucket_keys_for_chunk

    chunk = _resolved_chunk(
        "h1",
        [
            {
                "key": "concept:@خلق+عقل",
                "label": "خلق العقل",
                "type": "concept",
                "weight": 0.9,
                "parent": "concept:عقل",
                "broader": ["concept:معاد"],
            }
        ],
    )
    keys = bucket_keys_for_chunk(chunk)
    assert "concept:@خلق+عقل" in keys
    assert "concept:عقل" in keys      # child must meet its parent
    assert "concept:معاد" in keys     # curated siblings meet under a broader term


def test_summary_separates_what_proposed_a_pair_from_what_scored_it():
    """kitab never blocks, so it must not be reported as raising pairs."""
    docs = [
        Document(doc_id="d1", bab="باب واحد", kitab="كتاب واحد"),
        Document(doc_id="d2", bab="باب واحد", kitab="كتاب واحد"),
    ]
    pairs = generate(docs)
    assert pairs[0].blocked_by == "bab"
    assert "kitab" in pairs[0].reasons
    text = summarise(pairs, docs)
    assert "proposed by:" in text and "also scored on" in text


# ------------------------------------------------ structural, not exceptions
def test_a_compound_names_every_topic_in_it():
    """Structure is irrelevant; only which words name known concepts.

    Splitting on و was tried and corrupted real words -- `ولاة العدل` lost its
    conjunction and became `لاة العدل`. Here `في`, `على` and `قدر` contribute
    nothing simply because they name nothing.
    """
    from src.pipelines.resolver import decompose

    labels = lambda t: [pref for _, pref in decompose(t)]
    assert labels("الوسواس في الوضوء والصلاة") == ["الوسواس", "الوضوء", "الصلاة"]
    assert labels("الجزاء على قدر العقل") == ["الجزاء", "العقل"]
    assert labels("خلق العقل") == ["العقل"]
    # ولاة is a word, not a conjunction plus لاة. Stripping is only accepted
    # when what remains actually resolves, so no length heuristic can mangle it.
    assert labels("ولاة العدل") == ["العدل"]
    # A concept constituent must never reach the gazetteer: الحجة is an alias of
    # الإمام المهدي, and letting it through collapsed three kalam concepts onto him.
    assert labels("الحجة الباطنة") == []


def test_a_coordinated_mention_reaches_both_nodes(tmp_path: Path):
    from src.pipelines import resolve_pass

    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "marker": "10 -",
                "locator": "p12",
                "hadith": "مُبْتَلًى بِالْوُضُوءِ وَ الصَّلَاةِ",
                "mentions": [
                    {"text": "الوضوء والصلاة", "type": "concept", "salience": 0.8}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    resolve_pass.run(root=tmp_path)
    written = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    labels = {n["label"] for n in written["nodes"]}
    assert labels == {"الوضوء", "الصلاة"}


def test_a_curated_coordination_is_not_taken_apart():
    """الثواب والعقاب is a catalog alias of الجزاء الأخروي and stays whole."""
    nodes = resolve([_m("h1", "الثواب والعقاب")])
    assert [n.label for n in nodes.values()] == ["الجزاء الأخروي"]


def test_a_quoted_phrase_is_not_a_topic():
    """The 2:269 citation edge already carries it, and reaches the tafsir too."""
    from src.pipelines.grounding import ground_mentions

    matn = "وَالْعُقَلَاءُ هُمْ أُولُو الْأَلْبَابِ وَ الصَّبْرُ خَيْرٌ"
    quotes = [{"text": "وَمَا يَتَذَكَّرُ إِلَّا أُولُوا الْأَلْبَابِ", "kind": "quran"}]
    kept, rejected = ground_mentions(
        [
            {"text": "أولو الألباب", "type": "group", "evidence": "هُمْ أُولُو الْأَلْبَابِ"},
            # A single word is exempt: a hadith about الصبر that quotes a verse
            # mentioning الصبر is still about الصبر.
            {"text": "الصبر", "type": "concept", "evidence": "وَ الصَّبْرُ خَيْرٌ"},
        ],
        matn,
        quotes=quotes,
    )
    assert [m["text"] for m in kept] == ["الصبر"]
    assert rejected == [("أولو الألباب", "quoted scripture, not a topic")]


def test_a_node_key_and_its_type_always_agree():
    """A parent takes its own type, never the type of the child that reached it."""
    nodes = resolve(
        [_m("h1", "الوسواس في الوضوء"), _m("h2", "معاوية", "person"), _m("h3", "العقل")]
    )
    assert nodes
    for node in nodes.values():
        assert node.key.split(":", 1)[0] == node.type


def test_a_concept_never_collapses_onto_a_person():
    """الحجة is an alias of الإمام المهدي; three kalam concepts fell into him."""
    nodes = resolve(
        [
            _m("h1", "الحجة الباطنة"),
            _m("h2", "الحجة الظاهرة"),
            _m("h3", "إكمال الحجة"),
        ]
    )
    assert "الإمام المهدي" not in {n.label for n in nodes.values()}
    assert all(n.type == "concept" for n in nodes.values())


# ------------------------------------------------------ cross-page coverage
def test_continuation_page_contributes_its_own_mentions():
    """Page 2 of a spanning hadith sees matn page 1 never contained.

    The continuation branch passed only translations, so every observation made
    on the second half of a spanning narration was discarded -- the exact case
    the cross-page union exists for.
    """
    from src.pipelines.hadith_accumulator import consume_page

    page1 = "8- عَلِيُّ بْنُ مُحَمَّدٍ عَنْ إِبْرَاهِيمَ قَالَ إِنَّ الثَّوَابَ عَلَى قَدْرِ الْعَقْلِ\n"
    page2 = "وَ إِنَّ رَجُلًا مِنْ بَنِي إِسْرَائِيلَ كَانَ يَعْبُدُ اللَّهَ\n"
    items1 = [{
        "marker": "8-", "hadith": "x", "hadith_fa": "ف", "hadith_en": "e",
        "ravis": ["عَلِيُّ بْنُ مُحَمَّدٍ"],
        "mentions": [{"text": "الثواب", "type": "concept", "salience": 0.9,
                      "evidence": "إن الثواب على قدر العقل"}],
    }]
    items2 = [{
        "marker": "continuation", "hadith_fa": "ف٢", "hadith_en": "e2",
        "ravis": ["إِبْرَاهِيمُ"],
        "mentions": [{"text": "بني إسرائيل", "type": "group", "salience": 0.5,
                      "evidence": "رجلا من بني إسرائيل"}],
        "quotes": [{"text": "فاعتبروا يا أولي الأبصار", "kind": "quran"}],
    }]
    flushed, buf = consume_page("p11", page1, items1, None, page2)
    assert not flushed and buf is not None
    flushed, buf = consume_page("p12", page2, items2, buf, None)

    hadith = flushed[0]
    texts = [m["text"] for m in hadith["mentions"]]
    assert "الثواب" in texts, "page 1 mention lost"
    assert "بني إسرائيل" in texts, "page 2 mention never reached the buffer"
    # The quote spotted on page 2 resolves against the mushaf.
    assert "59:2" in hadith["quran_refs"]


def test_quoted_spans_become_citations():
    """`quotes` was collected and thrown away; the mushaf decides the verse."""
    from src.pipelines.hadith_accumulator import OpenHadith

    buf = OpenHadith(marker="5 -", page_start="p", page_end="p")
    buf.append_slice(
        "p",
        "إِنَّمَا قَالَ اللَّهُ فَاعْتَبِرُوا يا أُولِي الْأَبْصارِ",
        quotes=[{"text": "فاعتبروا يا أولي الأبصار", "kind": "quran"}],
    )
    assert buf.assemble()["quran_refs"] == ["59:2"]


# ---------------------------------------------------- one node space, really
def test_history_mentions_are_found_inside_events():
    """HistoryExtraction nests mentions per event; a root-only walk finds none."""
    from src.pipelines.resolve_pass import _mentions_of

    found = _mentions_of(
        {
            "events": [
                {
                    "event_title": "صفين",
                    "mentions": [
                        {"text": "الصبر", "type": "concept", "salience": 0.8},
                        {"text": "معاوية", "type": "person", "salience": 0.6},
                    ],
                }
            ]
        }
    )
    assert {m["text"] for m in found} == {"الصبر", "معاوية"}


def test_legacy_history_and_tafsir_shapes_still_resolve():
    from src.pipelines.resolve_pass import _mentions_of

    history = _mentions_of(
        {
            "events": [
                {
                    "event_title": "صفين",
                    "historical_concepts": ["الجهاد"],
                    "characters_involved": ["معاوية"],
                }
            ]
        }
    )
    by_text = {m["text"]: m for m in history}
    assert by_text["الجهاد"]["type"] == "concept"
    assert by_text["معاوية"]["type"] == "person"

    tafsir = _mentions_of({"ayah_anchor": "2:269", "core_concepts": ["الحكمة"]})
    assert [m["text"] for m in tafsir] == ["الحكمة"]


def test_all_three_pipelines_land_in_one_node_space(tmp_path: Path):
    """The claim the whole design rests on, checked end to end."""
    from src.pipelines import resolve_pass

    for name, payload in (
        ("hadith.json", {"marker": "1", "locator": "p1", "hadith": "الصبر",
                         "mentions": [{"text": "الصبر", "type": "concept", "salience": 0.9}]}),
        ("tafsir.json", {"locator": "t1", "hadith": "",
                         "mentions": [{"text": "الصبر", "type": "concept", "salience": 0.8}]}),
        ("history.json", {"locator": "h1", "hadith": "",
                          "events": [{"event_title": "x",
                                      "mentions": [{"text": "الصبر", "type": "concept",
                                                    "salience": 0.7}]}]}),
    ):
        (tmp_path / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    stats = resolve_pass.run(root=tmp_path)
    assert stats["documents"] == 3
    keys = {
        json.loads((tmp_path / n).read_text(encoding="utf-8"))["nodes"][0]["key"]
        for n in ("hadith.json", "tafsir.json", "history.json")
    }
    assert len(keys) == 1, "the three pipelines did not share a node"


# ------------------------------------------------- write-back after resolve
def test_retyped_mentions_reach_the_payload_not_just_the_node_table(tmp_path: Path):
    """The node table can be right while the payload the runtime reads is empty.

    Resolution retypes `place:الجنة` to `concept:الجنة`. Deriving assignments
    afterwards from a (type, surface) index looked up the ORIGINAL type, missed,
    and wrote the narration back with no nodes at all -- while `nodes.json`
    showed the correct label and a correct df of 2. Phase 2 and the export read
    the payload, so that hadith had nothing.
    """
    from src.pipelines import resolve_pass

    book = tmp_path / "hadith"
    book.mkdir()
    (book / "a.json").write_text(
        json.dumps(
            {
                "marker": "6 -",
                "locator": "p11",
                "hadith": "مَنْ كَانَ عَاقِلًا دَخَلَ الْجَنَّةَ",
                "mentions": [{"text": "الجنة", "type": "place", "salience": 0.6}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (book / "b.json").write_text(
        json.dumps(
            {
                "marker": "7 -",
                "locator": "p12",
                "hadith": "وَ الْجَنَّةُ دَارُ الْمُتَّقِينَ",
                "mentions": [{"text": "الجنة", "type": "concept", "salience": 0.8}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    resolve_pass.run(root=tmp_path)

    written = [
        json.loads((book / name).read_text(encoding="utf-8")) for name in ("a.json", "b.json")
    ]
    keys = [[n["key"] for n in payload["nodes"]] for payload in written]
    assert keys[0] and keys[1], "a retyped mention was dropped from the payload"
    assert keys[0] == keys[1], "both narrations must land on one node"
    assert written[0]["nodes"][0]["type"] == "concept"


def test_folded_and_merged_mentions_still_reach_the_payload(tmp_path: Path):
    """Keys move twice after clustering: compound folds and entity merges."""
    from src.pipelines import resolve_pass

    book = tmp_path / "hadith"
    book.mkdir()
    (book / "a.json").write_text(
        json.dumps(
            {
                "marker": "1", "locator": "p1",
                "hadith": "صَدِيقُ كُلِّ امْرِئٍ عَقْلُهُ وَ زُرَارَةُ بْنُ أَعْيَنَ",
                "mentions": [
                    {"text": "عقل المرء", "type": "concept", "salience": 0.9},
                    {"text": "زرارة بن أعين", "type": "person", "salience": 0.5},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (book / "b.json").write_text(
        json.dumps(
            {
                "marker": "2", "locator": "p2",
                "hadith": "قَالَ زُرَارَةُ وَ الْعَقْلُ",
                "mentions": [
                    {"text": "العقل", "type": "concept", "salience": 0.9},
                    {"text": "زرارة", "type": "person", "salience": 0.5},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    resolve_pass.run(root=tmp_path)

    a = json.loads((book / "a.json").read_text(encoding="utf-8"))
    b = json.loads((book / "b.json").read_text(encoding="utf-8"))
    # عقل المرء folded into العقل; both must carry the surviving key.
    assert set(n["key"] for n in a["nodes"]) == set(n["key"] for n in b["nodes"])
    assert len(a["nodes"]) == 2


# ------------------------------------------------------------ grounding scope
def test_page_level_remap_does_not_ground_away_a_spanning_mention():
    """`hadith` on a page item is only this page's fragment.

    Grounding there rejected any mention whose evidence sits on the next page,
    and the accumulator could never get it back.
    """
    from src.pipelines.ontology import remap_hadith_payload

    payload = remap_hadith_payload(
        {
            "hadiths": [
                {
                    "hadith": "أَوَّلُ الْمَتْنِ فَقَطْ",
                    "ravis": [],
                    "mentions": [
                        {"text": "الثواب", "type": "concept",
                         "evidence": "إِنَّ الثَّوَابَ عَلَى قَدْرِ الْعَقْلِ"},
                    ],
                }
            ]
        }
    )
    assert [m["text"] for m in payload["hadiths"][0]["mentions"]] == ["الثواب"]


def test_assembled_payload_is_still_grounded():
    from src.pipelines.ontology import remap_hadith_payload

    payload = remap_hadith_payload(
        {
            "hadith": "قُلْتُ لَهُ مَا الْعَقْلُ قَالَ مَا عُبِدَ بِهِ الرَّحْمَنُ",
            "ravis": [],
            "mentions": [
                {"text": "العقل", "type": "concept", "evidence": "ما عبد به الرحمن"},
                {"text": "الصبر", "type": "concept", "evidence": "و أمرهم بالصبر"},
            ],
        }
    )
    assert [m["text"] for m in payload["mentions"]] == ["العقل"]


def test_narrator_typed_as_a_concept_is_still_filtered():
    """Gating the filter on the declared type let the speaker in sideways."""
    from src.pipelines.grounding import ground_mentions

    kept, rejected = ground_mentions(
        [{"text": "أبو عبد الله", "type": "concept", "evidence": "عن أبي عبد الله"}],
        "عَنْ أَبِي عَبْدِ اللَّهِ ع قَالَ الْعَقْلُ",
        ["أَبُو عَبْدِ اللَّهِ (ع)"],
    )
    assert kept == []
    assert rejected == [("أبو عبد الله", "narrator, not a subject")]


def test_declared_type_wins_for_a_dual_listed_name():
    nodes = resolve([_m("a", "الشيطان", "person")])
    assert next(iter(nodes.values())).type == "person"
    # And a type nothing knows still gets corrected.
    nodes = resolve([_m("b", "الجنة", "place")])
    assert next(iter(nodes.values())).type == "concept"


def test_a_later_page_can_improve_an_earlier_mention():
    from src.pipelines.hadith_accumulator import OpenHadith

    buf = OpenHadith(marker="8-", page_start="p11", page_end="p11")
    buf.append_slice("p11", "أول", mentions=[
        {"text": "العقل", "type": "concept", "salience": 0.3, "evidence": ""}
    ])
    buf.append_slice("p12", "آخر", mentions=[
        {"text": "العقل", "type": "concept", "salience": 0.9, "evidence": "ما عبد به الرحمن"}
    ])
    assert len(buf.mentions_seed) == 1
    assert buf.mentions_seed[0]["salience"] == 0.9
    assert buf.mentions_seed[0]["evidence"] == "ما عبد به الرحمن"


# --------------------------------------------------------- tuning guardrails
def test_compound_threshold_scales_with_corpus_size():
    from src.pipelines.resolver import compound_threshold

    assert compound_threshold(30) == 2
    assert compound_threshold(15_000) > 2


def test_a_very_common_short_name_is_not_absorbed():
    """أبو محمد used constantly is several men, not one referred to briefly."""
    mentions = [_m(f"d{i}", "أبو محمد", "person") for i in range(40)]
    mentions += [_m("x", "أبو محمد الرازي", "person")]
    nodes = resolve(mentions)
    labels = {n.label for n in nodes.values()}
    assert "أبو محمد" in labels and "أبو محمد الرازي" in labels


# --------------------------------------------------------------- quran tiers
def test_root_tier_finds_a_verse_quoted_with_different_wording():
    """al-Kafi reads يَتَذَكَّرُ where 2:269 has يَذَّكَّرُ; both are the root ذكر."""
    if not match_quran_roots("وَ مَا يَتَذَكَّرُ إِلَّا أُولُوا الْأَلْبابِ"):
        import pytest

        pytest.skip("Qur'an corpus not available")
    assert "2:269" in match_quran_roots("وَ مَا يَتَذَكَّرُ إِلَّا أُولُوا الْأَلْبابِ")


def test_tiers_combine_without_duplicating():
    refs = match_quran("وَ مَا يَتَذَكَّرُ إِلَّا أُولُوا الْأَلْبابِ")
    if not refs:
        import pytest

        pytest.skip("Qur'an corpus not available")
    assert len(refs) == len(set(refs))


def test_ordinary_arabic_matches_no_verse():
    assert match_quran("صَدِيقُ كُلِّ امْرِئٍ عَقْلُهُ وَ عَدُوُّهُ جَهْلُهُ") == []
    assert match_quran("علي بن ابراهيم عن ابيه عن النوفلي عن السكوني") == []
