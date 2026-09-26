# shiadata-graph

Resumable ETL for Shia classical texts: parse the local corpus, extract structured
records, turn mentions into graph nodes, embed with OpenAI, canonicalise duplicate
hadiths, classify SUPPORTS / CONTRADICTS / EXCEPTS edges, and export Neo4j-ready JSONL.

## Node identity

Phase 1 writes **mentions** (what this narration is about). It does not decide
which mentions are the same idea.

1. **Unresolved nodes.** Each distinct mention stays its own node. Nothing is
   deleted or folded into another label. A narrower wording may be linked as a
   **child** of a broader one (`عقل المرء` under `العقل`). Both nodes remain.
2. **Pre-phase 2 merge.** Those unresolved nodes are embedded **with the hadith
   (or evidence span) that produced them**, not as bare labels. Nodes that mean
   the same thing in that context are merged (`محبة أهل البيت` and
   `إرادة أهل البيت`). The original wordings stay as children of the merged
   node; the merge does not erase them.
3. **Phase 2** embeds hadiths, confirms duplicate narrations, and classifies
   SUPPORTS / CONTRADICTS / EXCEPTS. It does not decide node synonymy.

`resolve-nodes` today still clusters by catalog and morphology. The two-step
identity above is the workflow those clusters must follow; synonym merges belong
in the embedding step, not in per-page extraction.

## Agents (reuse these; do not open SDK clients in new phases)

```python
from src.state_manager import StateManager
from src.agents import GeminiAgent, EmbeddingAgent
from src.models import HadithExtraction  # or any Pydantic schema

state = StateManager()
gemini = GeminiAgent(state)
embeddings = EmbeddingAgent(state=state)

record = gemini.complete_structured(prompt, HadithExtraction, system="...")
vectors = embeddings.embed(["text"])
```

`GeminiAgent` owns round-robin `GOOGLE_API_KEY*` rotation, 429/quota cooldowns,
and `AllKeysExhausted` (exit code 2). Re-run the same CLI command to resume.

## Corpus

Books were copied from `shiadata-rag/data/raw_epubs` into `data/raw_epubs`.
Sources are Folklib-style `.txt` banners (`--- [جلد 1 - صفحه 1] ---` or
`--- [سوره 1 - آیات 1-5] ---`). EPUB is supported if you add `.epub` files later.

## Setup

```bash
cd shiadata-graph
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Keys are read from `../shiadata-rag/.env` then local `.env`. See `.env.example`.

## Commands

```bash
python main.py list-books
python main.py run-phase1 --book al-kafi --limit 3
python main.py run-phase1 --book al-mizan --limit 2
python main.py run-phase1 --book waqat-siffin --limit 1
python main.py run-phase2 --book al-kafi
python main.py export-neo4j
python main.py reset-book --book hadith --yes
python main.py status --book al-kafi
```

If every Gemini key is cooling, the process saves SQLite state and exits 2.
Resume the next day with the same command.

## Tests

```bash
pytest
```
