"""
modal_processors_demo.py
────────────────────────
Demonstrates individual multimodal processors from RAGAnything:
  • ImageModalProcessor  — describe images, extract entities
  • TableModalProcessor  — parse tables, extract structured data
  • EquationProcessor    — understand math / LaTeX

Run:
    python modal_processors_demo.py
"""

import asyncio
import os
import logging
from pathlib import Path

from gemini_adapters import (
    configure,
    gemini_llm_complete,
    gemini_vision_complete,
    get_embedding_func,
)
from rag_gemini_main import build_rag

log = logging.getLogger("modal_demo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

configure()  # reads GEMINI_API_KEY from env


# ─────────────────────────────────────────────────────────────────────────────
#  Image Processor Demo
# ─────────────────────────────────────────────────────────────────────────────
async def demo_image_processor():
    """Process a synthetic image description."""
    try:
        from raganything.modalprocessors import ImageModalProcessor
    except ImportError:
        log.warning("raganything not installed — skipping image processor demo")
        return

    rag = build_rag()
    await rag.initialize_storages()

    processor = ImageModalProcessor(
        lightrag=rag,
        modal_caption_func=gemini_vision_complete,
    )

    image_content = {
        "img_path": "figures/architecture_diagram.png",
        "img_caption": [
            "Figure 2: System architecture of RAG-Anything showing the "
            "three-layer design: document parsing, knowledge graph construction, "
            "and hybrid retrieval."
        ],
        "img_footnote": [
            "Grey nodes = text entities; Blue nodes = image entities; "
            "Green edges = cross-modal relationships"
        ],
    }

    log.info("🖼️   Processing image content...")
    description, entity_info = await processor.process_multimodal_content(
        modal_content=image_content,
        content_type="image",
        file_path="research_paper.pdf",
        entity_name="Architecture Diagram Figure 2",
    )
    print("\n── Image Processor Output ──────────────────────────────────────")
    print(f"Description preview: {description[:400]}...")
    print(f"Entity info: {entity_info}")
    return description, entity_info


# ─────────────────────────────────────────────────────────────────────────────
#  Table Processor Demo
# ─────────────────────────────────────────────────────────────────────────────
async def demo_table_processor():
    """Process a performance comparison table."""
    try:
        from raganything.modalprocessors import TableModalProcessor
    except ImportError:
        log.warning("raganything not installed — skipping table processor demo")
        return

    rag = build_rag()
    await rag.initialize_storages()

    processor = TableModalProcessor(
        lightrag=rag,
        modal_caption_func=gemini_llm_complete,
    )

    table_content = {
        "table_body": (
            "| Framework     | Recall@5 | MRR   | F1    | Modalities Supported |\n"
            "|---------------|----------|-------|-------|---------------------|\n"
            "| RAG-Anything  | 0.923    | 0.871 | 0.894 | Text, Image, Table  |\n"
            "| MMGraphRAG    | 0.887    | 0.832 | 0.858 | Text, Image         |\n"
            "| LightRAG      | 0.851    | 0.798 | 0.821 | Text                |\n"
            "| NaiveRAG      | 0.743    | 0.691 | 0.714 | Text                |\n"
            "| GPT-4o-mini   | 0.768    | 0.712 | 0.738 | Text, Image         |"
        ),
        "table_caption": ["Table 3: Retrieval performance on MuMuQA benchmark"],
        "table_footnote": [
            "All scores averaged over 3 runs",
            "Recall@5 = percentage of correct answers in top-5 retrieved passages",
        ],
    }

    log.info("📊  Processing table content...")
    description, entity_info = await processor.process_multimodal_content(
        modal_content=table_content,
        content_type="table",
        file_path="research_paper.pdf",
        entity_name="Performance Comparison Table 3",
    )
    print("\n── Table Processor Output ───────────────────────────────────────")
    print(f"Description preview: {description[:400]}...")
    print(f"Entity info: {entity_info}")
    return description, entity_info


# ─────────────────────────────────────────────────────────────────────────────
#  Equation Processor Demo
# ─────────────────────────────────────────────────────────────────────────────
async def demo_equation_processor():
    """Process LaTeX equations."""
    try:
        from raganything.modalprocessors import EquationModalProcessor
    except ImportError:
        log.warning("raganything not installed — skipping equation processor demo")
        return

    rag = build_rag()
    await rag.initialize_storages()

    processor = EquationModalProcessor(
        lightrag=rag,
        modal_caption_func=gemini_llm_complete,
    )

    equation_content = {
        "latex": (
            r"P(q | G) = \sum_{e \in E_q} \left[ "
            r"\alpha \cdot \text{sim}(q, e) + "
            r"(1-\alpha) \cdot \sum_{r \in \mathcal{N}(e)} w_r \cdot \text{sim}(q, r) "
            r"\right]"
        ),
        "equation_caption": [
            "Equation 1: Graph-augmented retrieval probability. "
            "E_q = entities related to query q, N(e) = neighbourhood of entity e, "
            "w_r = edge weight, alpha = balance parameter."
        ],
    }

    log.info("🔢  Processing equation content...")
    description, entity_info = await processor.process_multimodal_content(
        modal_content=equation_content,
        content_type="equation",
        file_path="research_paper.pdf",
        entity_name="Graph Retrieval Probability Equation 1",
    )
    print("\n── Equation Processor Output ────────────────────────────────────")
    print(f"Description preview: {description[:400]}...")
    print(f"Entity info: {entity_info}")
    return description, entity_info


# ─────────────────────────────────────────────────────────────────────────────
#  Run all demos
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 65)
    print("  RAG-Anything Modal Processors Demo  (Gemini-powered)")
    print("=" * 65)

    await demo_image_processor()
    await demo_table_processor()
    await demo_equation_processor()

    print("\n✅  All modal processor demos complete.")


if __name__ == "__main__":
    asyncio.run(main())
