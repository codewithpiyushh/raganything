"""
RAG-Anything + Gemini: End-to-End Multimodal Knowledge Graph RAG
================================================================
Full integration of RAG-Anything (HKUDS) with Google Gemini for:
  - Text, Images, Tables, Equations, Office Docs, PDFs
  - Multimodal Knowledge Graph construction
  - Hybrid retrieval (vector + graph)
  - Gemini LLM + Vision + Embeddings

Requirements:
    pip install raganything lightrag-hku google-generativeai python-dotenv

Usage:
    python rag_gemini_main.py --mode process --file documents/sample.pdf
    python rag_gemini_main.py --mode query  --question "What does the table show?"
    python rag_gemini_main.py --mode demo   (runs full demo with synthetic docs)
"""

import asyncio
import os
import base64
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from lightrag.utils import EmbeddingFunc

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag-gemini")

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
WORKING_DIR      = "./rag_storage"
OUTPUT_DIR       = "./parsed_output"
DOCUMENTS_DIR    = "./documents"
DEFAULT_FILE     = "C:/Users/hp/Desktop/coding/files/Internship.pdf"   # ← Put your file path here, e.g. "C:/Users/hp/Documents/paper.pdf"
EMBEDDING_DIM    = 768          # Gemini text-embedding-004 → 768-d
EMBEDDING_MODEL  = "models/text-embedding-004"
LLM_MODEL        = "gemini-2.0-flash"
VISION_MODEL     = "gemini-2.0-flash"   # multimodal — handles images natively

# ── Gemini client setup ───────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  LLM FUNCTION  (text generation)
# ═══════════════════════════════════════════════════════════════════════════════
async def gemini_llm_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: list = [],
    **kwargs,
) -> str:
    """Async wrapper around Gemini text generation."""
    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        system_instruction=system_prompt or "You are a helpful research assistant.",
    )

    # Build conversation history
    history = []
    for msg in history_messages:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)

    try:
        response = await asyncio.to_thread(chat.send_message, prompt)
        return response.text
    except Exception as e:
        log.error(f"LLM error: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  VISION FUNCTION  (image understanding)
# ═══════════════════════════════════════════════════════════════════════════════
async def gemini_vision_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: list = [],
    image_data: Optional[str] = None,   # base64-encoded image
    messages: Optional[list] = None,    # pre-built multimodal messages
    **kwargs,
) -> str:
    """Async Gemini vision wrapper — handles images + text."""
    model = genai.GenerativeModel(
        model_name=VISION_MODEL,
        system_instruction=system_prompt or "You are a multimodal document analyst.",
    )

    # Case 1: caller already built a messages list
    if messages:
        parts = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item["type"] == "text":
                        parts.append(item["text"])
                    elif item["type"] == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64 = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            parts.append(
                                {"mime_type": mime, "data": base64.b64decode(b64)}
                            )
            elif isinstance(msg.get("content"), str):
                parts.append(msg["content"])
        try:
            response = await asyncio.to_thread(model.generate_content, parts)
            return response.text
        except Exception as e:
            log.error(f"Vision (messages) error: {e}")
            raise

    # Case 2: separate image_data + prompt
    if image_data:
        img_bytes = base64.b64decode(image_data)
        parts = [{"mime_type": "image/jpeg", "data": img_bytes}, prompt]
    else:
        parts = [prompt]

    try:
        response = await asyncio.to_thread(model.generate_content, parts)
        return response.text
    except Exception as e:
        log.error(f"Vision error: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  EMBEDDING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
async def gemini_embed_func(texts: list[str]) -> list[list[float]]:
    """Batch text embedding using Gemini text-embedding-004 (768-d)."""
    embeddings = []
    for text in texts:
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        except Exception as e:
            log.error(f"Embedding error for text snippet: {e}")
            embeddings.append([0.0] * EMBEDDING_DIM)
    return embeddings


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  RAGAnything BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def build_rag() -> "RAGAnything":  # noqa: F821
    """Construct a RAGAnything instance wired to Gemini models."""
    from raganything import RAGAnything, RAGAnythingConfig

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR,  exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        # Parser: use 'mineru' for best quality (requires: pip install mineru)
        # Fallback to 'docling' if MinerU is not installed
        parser="paddleocr",
        parse_method="auto",          # auto | ocr | txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=gemini_llm_func,
        vision_model_func=gemini_vision_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=gemini_embed_func,
        ),
    )
    return rag


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  PROCESS SINGLE DOCUMENT
# ═══════════════════════════════════════════════════════════════════════════════
async def process_document(file_path: str):
    """Ingest one document into the knowledge graph."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    log.info(f"📄  Processing: {file_path}")
    rag = build_rag()
    await rag.finalize_storages()

    await rag.process_document_complete(
        file_path=file_path,
        output_dir=OUTPUT_DIR,
        parse_method="auto",
    )
    log.info("✅  Document processed and indexed into knowledge graph.")


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  PROCESS ENTIRE FOLDER
# ═══════════════════════════════════════════════════════════════════════════════
async def process_folder(folder_path: str):
    """Recursively ingest all supported documents from a folder."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    log.info(f"📁  Processing folder: {folder_path}")
    rag = build_rag()
    await rag.initialize_storages()

    await rag.process_folder_complete(
        folder_path=folder_path,
        output_dir=OUTPUT_DIR,
        recursive=True,
    )
    log.info("✅  All documents processed.")


# ═══════════════════════════════════════════════════════════════════════════════
#  7.  QUERY
# ═══════════════════════════════════════════════════════════════════════════════
async def query(question: str, mode: str = "hybrid") -> str:
    """
    Query the knowledge graph.

    mode options:
        naive   – simple vector similarity
        local   – entity-focused graph traversal
        global  – community-level summarisation
        hybrid  – local + global (recommended)
        mix     – naive + graph
    """
    log.info(f"🔍  Query [{mode}]: {question}")
    rag = build_rag()
    await rag.initialize_storages()

    answer = await rag.aquery(question, param={"mode": mode})
    return answer


# ═══════════════════════════════════════════════════════════════════════════════
#  8.  DIRECT CONTENT INJECTION  (bypass parser)
# ═══════════════════════════════════════════════════════════════════════════════
async def inject_content_list(content_list: list, file_path: str = "injected"):
    """
    Insert pre-parsed multimodal content directly.
    Useful when you already have structured data (tables, image captions, etc.)

    content_list items follow MinerU's content_list schema:
        {"type": "text",      "text": "..."}
        {"type": "image",     "img_path": "...", "img_caption": [...]}
        {"type": "table",     "table_body": "...", "table_caption": [...]}
        {"type": "equation",  "latex": "..."}
    """
    log.info(f"💉  Injecting {len(content_list)} content blocks")
    rag = build_rag()
    await rag.initialize_storages()

    await rag.process_content_list(
        content_list=content_list,
        file_path=file_path,
        output_dir=OUTPUT_DIR,
    )
    log.info("✅  Content injected.")


# ═══════════════════════════════════════════════════════════════════════════════
#  9.  DEMO  — synthetic multimodal document
# ═══════════════════════════════════════════════════════════════════════════════
async def run_demo():
    """
    End-to-end demo using synthetic multimodal content:
      • Text passages
      • A markdown table
      • An image description (simulated)
      • A LaTeX equation
    Then runs several queries across all retrieval modes.
    """
    log.info("🚀  Starting RAG-Anything + Gemini DEMO")

    # ── synthetic content list ─────────────────────────────────────────────
    content_list = [
        {
            "type": "text",
            "text": (
                "RAG-Anything is an all-in-one multimodal Retrieval-Augmented "
                "Generation framework developed by HKUDS. It extends LightRAG "
                "to support documents containing text, images, tables, and "
                "mathematical equations. The system builds a multimodal "
                "knowledge graph that captures cross-modal relationships."
            ),
        },
        {
            "type": "table",
            "table_body": (
                "| Model        | MMQA Score | Latency (ms) | Modalities       |\n"
                "|--------------|------------|--------------|------------------|\n"
                "| RAG-Anything | 91.4       | 320          | Text+Image+Table |\n"
                "| LightRAG     | 78.2       | 210          | Text only        |\n"
                "| GPT-4o-mini  | 74.5       | 450          | Text+Image       |\n"
                "| Baseline RAG | 62.1       | 180          | Text only        |"
            ),
            "table_caption": ["Table 1: Performance comparison on MMQA benchmark"],
            "table_footnote": ["Latency measured on a single A100 GPU, batch size 1"],
        },
        {
            "type": "image",
            "img_path": "figure1_architecture.png",  # path (may not exist in demo)
            "img_caption": [
                "Figure 1: RAG-Anything architecture showing the multimodal "
                "pipeline from document ingestion to knowledge graph construction "
                "and hybrid retrieval."
            ],
            "img_footnote": ["Dashed boxes indicate optional components"],
        },
        {
            "type": "text",
            "text": (
                "The framework achieves superior performance by combining two "
                "retrieval strategies: vector-based semantic similarity search "
                "and graph-based structural traversal through the knowledge graph. "
                "Gemini models power both the language understanding and the vision "
                "components, while Gemini text-embedding-004 provides dense "
                "768-dimensional embeddings for all modalities."
            ),
        },
        {
            "type": "equation",
            "latex": (
                r"Score(q, d) = \alpha \cdot \text{VectorSim}(q, d) "
                r"+ (1-\alpha) \cdot \text{GraphScore}(q, d)"
            ),
            "equation_caption": ["Hybrid retrieval scoring formula"],
        },
        {
            "type": "text",
            "text": (
                "Google Gemini 2.0 Flash is used as the primary LLM for entity "
                "extraction and answer generation. The embedding model is "
                "text-embedding-004, which produces 768-dimensional vectors "
                "optimized for retrieval tasks. The vision model processes "
                "images and figures directly from documents."
            ),
        },
        {
            "type": "table",
            "table_body": (
                "| Component          | Model Used                  | Dimensions |\n"
                "|--------------------|-----------------------------|-----------|\n"
                "| LLM                | gemini-2.0-flash            | N/A        |\n"
                "| Vision             | gemini-2.0-flash (vision)   | N/A        |\n"
                "| Text Embeddings    | text-embedding-004          | 768        |\n"
                "| Knowledge Graph    | NetworkX (in-memory)        | N/A        |"
            ),
            "table_caption": ["Table 2: System components and models"],
            "table_footnote": [],
        },
    ]

    # ── inject content ─────────────────────────────────────────────────────
    await inject_content_list(
        content_list=content_list,
        file_path="demo_multimodal_paper.pdf",
    )

    # ── queries across all modes ───────────────────────────────────────────
    questions = [
        ("What is RAG-Anything and how does it differ from standard RAG?", "hybrid"),
        ("Which model scored the highest on the MMQA benchmark?",          "local"),
        ("What embedding model and dimensions are used in this system?",   "local"),
        ("Describe the hybrid retrieval scoring formula.",                  "global"),
        ("Compare RAG-Anything and LightRAG in terms of modalities.",      "hybrid"),
    ]

    results = {}
    print("\n" + "═" * 70)
    print("  RAG-ANYTHING + GEMINI  —  DEMO QUERY RESULTS")
    print("═" * 70)

    for question, mode in questions:
        print(f"\n❓  [{mode.upper()}] {question}")
        try:
            answer = await query(question, mode=mode)
            results[question] = answer
            # Pretty-print wrapped
            print(f"💬  {answer[:600]}{'…' if len(answer) > 600 else ''}")
        except Exception as e:
            log.error(f"Query failed: {e}")
            results[question] = f"ERROR: {e}"

    # ── save results ───────────────────────────────────────────────────────
    os.makedirs("./demo_results", exist_ok=True)
    results_path = "./demo_results/query_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "═" * 70)
    log.info(f"✅  Demo complete. Results saved to {results_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Entry-point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="RAG-Anything + Gemini: End-to-End Multimodal KG RAG"
    )
    parser.add_argument(
        "--mode",
        choices=["process", "process_folder", "query", "inject", "demo"],
        default="demo",
        help="Operation mode",
    )
    parser.add_argument("--file",     help="Path to document file (process mode)")
    parser.add_argument("--folder",   help="Path to documents folder (process_folder mode)")
    parser.add_argument("--question", help="Question to answer (query mode)")
    parser.add_argument(
        "--retrieval_mode",
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="Retrieval strategy (query mode)",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        asyncio.run(run_demo())

    elif args.mode == "process":
        file_path = args.file or DEFAULT_FILE
        if not file_path:
            parser.error("--file is required for process mode (or set DEFAULT_FILE in code)")
        asyncio.run(process_document(file_path))

    elif args.mode == "process_folder":
        folder = args.folder or DOCUMENTS_DIR
        asyncio.run(process_folder(folder))

    elif args.mode == "query":
        if not args.question:
            parser.error("--question is required for query mode")
        answer = asyncio.run(query(args.question, mode=args.retrieval_mode))
        print(f"\n💬  Answer:\n{answer}")

    elif args.mode == "inject":
        # Example: inject a simple text + table
        sample = [
            {"type": "text", "text": "Injected via CLI."},
            {
                "type": "table",
                "table_body": "| A | B |\n|---|---|\n| 1 | 2 |",
                "table_caption": ["Sample table"],
                "table_footnote": [],
            },
        ]
        asyncio.run(inject_content_list(sample, file_path="cli_injection.pdf"))


if __name__ == "__main__":
    main()
