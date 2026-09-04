# shiadata-graph

Resumable ETL for Shia classical texts: parse the local corpus, extract structured
records with Gemini 3.5 Flash, embed with OpenAI, canonicalise duplicate hadiths,
classify SUPPORTS / CONTRADICTS / EXCEPTS edges, and export Neo4j-ready JSONL.

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
