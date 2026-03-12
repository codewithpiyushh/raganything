# RAG-Anything + Gemini  
**End-to-End Multimodal Knowledge Graph RAG**

> Combines [RAG-Anything (HKUDS)](https://github.com/HKUDS/RAG-Anything) with  
> Google Gemini for LLM, vision understanding, and text embeddings.

---

## What this project does

| Feature | Detail |
|---|---|
| Document formats | PDF, DOCX, PPTX, XLSX, images (PNG/JPG/BMP), Markdown, plain text |
| Modality support | Text passages · Tables · Images/Figures · LaTeX equations |
| Knowledge graph | Auto-extracted multimodal entity/relation graph (LightRAG-powered) |
| Retrieval modes | `naive` · `local` · `global` · `hybrid` · `mix` |
| LLM / Vision | Gemini 2.0 Flash (text + vision) |
| Embeddings | `text-embedding-004` → 768-dimensional vectors |

---

## File structure

```
rag_anything_gemini/
├── rag_gemini_main.py        ← main pipeline (process, query, demo)
├── gemini_adapters.py        ← Gemini LLM / vision / embedding wrappers
├── modal_processors_demo.py  ← per-modality processor demos
├── kg_inspector.py           ← inspect & export the knowledge graph
├── tests.py                  ← test suite
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Install dependencies

```bash
pip install raganything google-generativeai python-dotenv networkx numpy

# Recommended: install MinerU for best document parsing quality
pip install mineru

# For Office documents (.docx, .pptx, .xlsx) also install LibreOffice:
#   macOS:  brew install libreoffice
#   Ubuntu: sudo apt install libreoffice
```

### 2. Set your Gemini API key

```bash
# Option A: environment variable
export GEMINI_API_KEY="your-key-here"

# Option B: .env file
cp .env.example .env
# then edit .env and add your key
```

Get a free API key at: https://aistudio.google.com/app/apikey

---

## Quick start

### Run the built-in demo (no documents needed)

```bash
python rag_gemini_main.py --mode demo
```

This injects a synthetic multimodal document (text + tables + image captions + equations)  
into the knowledge graph, then runs 5 queries across different retrieval modes.

### Process your own document

```bash
python rag_gemini_main.py --mode process --file path/to/document.pdf
```

Supported: `.pdf` `.docx` `.pptx` `.xlsx` `.doc` `.png` `.jpg` `.md` `.txt`

### Process an entire folder

```bash
python rag_gemini_main.py --mode process_folder --folder ./my_docs
```

### Query

```bash
python rag_gemini_main.py --mode query \
    --question "Summarise the key findings" \
    --retrieval_mode hybrid
```

Retrieval modes:

| Mode | Best for |
|---|---|
| `naive` | Simple similarity, fast |
| `local` | Entity-level questions ("What is X?") |
| `global` | Big-picture summaries |
| `hybrid` | **Recommended** — combines local + global |
| `mix` | Combines naive + graph |

---

## Programmatic usage

```python
import asyncio
from rag_gemini_main import process_document, query

async def main():
    # Index a PDF
    await process_document("report.pdf")

    # Query it
    answer = await query(
        "What does Table 3 show about model performance?",
        mode="hybrid"
    )
    print(answer)

asyncio.run(main())
```

### Direct content injection (no parser needed)

```python
from rag_gemini_main import inject_content_list

content = [
    {"type": "text",  "text": "My custom knowledge passage."},
    {
        "type": "table",
        "table_body": "| A | B |\n|---|---|\n| 1 | 2 |",
        "table_caption": ["My table"],
        "table_footnote": [],
    },
    {
        "type": "image",
        "img_path": "diagram.png",
        "img_caption": ["Architecture diagram"],
        "img_footnote": [],
    },
]

asyncio.run(inject_content_list(content, file_path="my_source.pdf"))
```

---

## Inspect the knowledge graph

```bash
# Print stats + sample entities/relations
python kg_inspector.py --working_dir ./rag_storage

# Export to GraphML (open in Gephi or Cytoscape)
python kg_inspector.py --export_graphml

# Export summary JSON
python kg_inspector.py --export_json
```

---

## Run tests

```bash
# All tests
python tests.py

# Single section
python tests.py --section connectivity
python tests.py --section embeddings
python tests.py --section rag_build
python tests.py --section injection
python tests.py --section query
python tests.py --section kg_inspector
```

---

## Models used

| Role | Model | Notes |
|---|---|---|
| Text generation | `gemini-2.0-flash` | Knowledge graph extraction + answering |
| Vision | `gemini-2.0-flash` | Image / figure understanding |
| Embeddings | `text-embedding-004` | 768-d, optimised for retrieval |

Override via environment variables:
```bash
export GEMINI_LLM_MODEL=gemini-2.0-flash-lite   # cheaper
export GEMINI_EMBED_DIM=768
```

---

## Architecture

```
Document (PDF / DOCX / image / …)
        │
        ▼
  [MinerU / Docling Parser]
        │  content_list
        ▼
  ┌─────────────────────────────────┐
  │    Modality Routers             │
  │  Text ── Table ── Image ── Eq.  │
  └──────────────┬──────────────────┘
                 │  Gemini Vision / LLM descriptions
                 ▼
  ┌─────────────────────────────────┐
  │  Multimodal Knowledge Graph     │
  │  (LightRAG / NetworkX)          │
  │  entities + cross-modal edges   │
  └──────────────┬──────────────────┘
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
  Vector Store          Graph Store
  (NanoVectorDB)        (NetworkX)
       └─────────┬──────────┘
                 │  Hybrid retrieval
                 ▼
          [Gemini 2.0 Flash]
                 │
                 ▼
            Final Answer
```

---

## Troubleshooting

**`ModuleNotFoundError: raganything`**  
→ `pip install raganything`

**`MinerU not found`**  
→ `pip install mineru` or set `parser="docling"` in `build_rag()` config

**Rate limit errors from Gemini**  
→ Add `await asyncio.sleep(1)` between documents, or use a paid-tier key

**Office documents not parsing**  
→ Install LibreOffice (see Setup section above)

**Empty knowledge graph after processing**  
→ Check that `GEMINI_API_KEY` is set and the document isn't empty or image-only without OCR
