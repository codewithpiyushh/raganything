"""
tests.py
────────
Unit / integration tests for the RAG-Anything + Gemini stack.

Run all tests:
    python tests.py

Run specific section:
    python tests.py --section embeddings
    python tests.py --section llm
    python tests.py --section rag
"""

import asyncio
import os
import sys
import argparse
import logging
import time
from typing import Callable

log = logging.getLogger("tests")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
#  Test runner helpers
# ─────────────────────────────────────────────────────────────────────────────
PASS  = "✅  PASS"
FAIL  = "❌  FAIL"
SKIP  = "⏭️   SKIP"

results: list[tuple[str, str, str]] = []   # (section, name, status)


def record(section: str, name: str, passed: bool, reason: str = ""):
    status = PASS if passed else f"{FAIL} — {reason}"
    results.append((section, name, status))
    icon = "✅" if passed else "❌"
    print(f"  {icon}  {name}" + (f"  [{reason}]" if reason and not passed else ""))


async def _run_test(section: str, name: str, coro):
    try:
        result = await coro
        record(section, name, True)
        return result
    except Exception as exc:
        record(section, name, False, str(exc)[:120])
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  1. Gemini API connectivity
# ─────────────────────────────────────────────────────────────────────────────
async def test_gemini_connectivity():
    print("\n── Section: Gemini API Connectivity ────────────────────────────")
    from gemini_adapters import configure, gemini_llm_complete, gemini_embed

    configure()

    # LLM
    async def _llm():
        resp = await gemini_llm_complete("Reply with one word: hello")
        assert isinstance(resp, str) and len(resp) > 0, "Empty response"
        return resp
    await _run_test("connectivity", "Gemini LLM (text)", _llm())

    # Embeddings
    async def _embed():
        vecs = await gemini_embed(["test embedding"])
        assert len(vecs) == 1 and len(vecs[0]) == 768, f"Expected 768-d, got {len(vecs[0])}"
        return vecs
    await _run_test("connectivity", "Gemini Embeddings (768-d)", _embed())

    # Batch embeddings
    async def _embed_batch():
        texts = ["alpha", "beta", "gamma", "delta"]
        vecs = await gemini_embed(texts)
        assert len(vecs) == 4
        assert all(len(v) == 768 for v in vecs), "Dimension mismatch in batch"
        return vecs
    await _run_test("connectivity", "Gemini Embeddings (batch)", _embed_batch())


# ─────────────────────────────────────────────────────────────────────────────
#  2. Vision model
# ─────────────────────────────────────────────────────────────────────────────
async def test_vision():
    print("\n── Section: Gemini Vision ──────────────────────────────────────")
    from gemini_adapters import configure, gemini_vision_complete
    configure()

    # Text-only prompt (vision model handles text too)
    async def _text_prompt():
        resp = await gemini_vision_complete("What is 2+2? Reply with just the number.")
        assert "4" in resp, f"Expected '4' in response, got: {resp}"
        return resp
    await _run_test("vision", "Vision model — text-only prompt", _text_prompt())

    # Pre-formatted messages
    async def _messages_format():
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Say 'vision ok'"}]}
        ]
        resp = await gemini_vision_complete("", messages=messages)
        assert isinstance(resp, str) and len(resp) > 0
        return resp
    await _run_test("vision", "Vision model — messages format", _messages_format())


# ─────────────────────────────────────────────────────────────────────────────
#  3. RAGAnything import & build
# ─────────────────────────────────────────────────────────────────────────────
async def test_rag_build():
    print("\n── Section: RAGAnything Build ──────────────────────────────────")

    try:
        from raganything import RAGAnything, RAGAnythingConfig
    except ImportError:
        record("rag_build", "raganything import", False, "pip install raganything")
        return

    record("rag_build", "raganything import", True)

    try:
        from rag_gemini_main import build_rag
        rag = build_rag()
        record("rag_build", "RAGAnything instantiation", True)
    except Exception as exc:
        record("rag_build", "RAGAnything instantiation", False, str(exc)[:120])
        return

    # Storage initialisation
    async def _init():
        await rag.initialize_storages()
        return True
    await _run_test("rag_build", "Storage initialisation", _init())


# ─────────────────────────────────────────────────────────────────────────────
#  4. Content injection (no document parser needed)
# ─────────────────────────────────────────────────────────────────────────────
async def test_content_injection():
    print("\n── Section: Content Injection ──────────────────────────────────")

    try:
        from raganything import RAGAnything
    except ImportError:
        record("injection", "raganything import", False, "pip install raganything")
        return

    from rag_gemini_main import inject_content_list

    sample_content = [
        {"type": "text", "text": "The quick brown fox jumps over the lazy dog."},
        {
            "type": "table",
            "table_body": "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
            "table_caption": ["Test table"],
            "table_footnote": [],
        },
    ]

    async def _inject():
        await inject_content_list(sample_content, file_path="test_injection.pdf")
        return True
    await _run_test("injection", "Text + table injection", _inject())


# ─────────────────────────────────────────────────────────────────────────────
#  5. Query after injection
# ─────────────────────────────────────────────────────────────────────────────
async def test_query():
    print("\n── Section: Query ──────────────────────────────────────────────")

    try:
        from raganything import RAGAnything
    except ImportError:
        record("query", "raganything import", False, "pip install raganything")
        return

    from rag_gemini_main import query

    modes = ["naive", "local", "global", "hybrid"]
    for mode in modes:
        async def _q(m=mode):
            answer = await query("What is the quick brown fox?", mode=m)
            assert isinstance(answer, str) and len(answer) > 0, "Empty answer"
            return answer
        await _run_test("query", f"Query mode={mode}", _q())


# ─────────────────────────────────────────────────────────────────────────────
#  6. KG Inspector
# ─────────────────────────────────────────────────────────────────────────────
async def test_kg_inspector():
    print("\n── Section: KG Inspector ───────────────────────────────────────")

    async def _load():
        from kg_inspector import load_graph_data
        data = load_graph_data("./rag_storage")
        assert isinstance(data, dict)
        assert "entities" in data
        return data
    data = await _run_test("kg_inspector", "Load graph data", _load())

    if data:
        async def _stats():
            from kg_inspector import print_statistics
            print_statistics(data)
            return True
        await _run_test("kg_inspector", "Print statistics", _stats())

        async def _export():
            from kg_inspector import export_summary_json
            path = export_summary_json(data, output_path="/tmp/kg_test_summary.json")
            import os
            assert os.path.exists("/tmp/kg_test_summary.json")
            return path
        await _run_test("kg_inspector", "Export summary JSON", _export())


# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary():
    print("\n" + "═" * 60)
    print("  TEST SUMMARY")
    print("═" * 60)
    passed = sum(1 for _, _, s in results if s.startswith("✅"))
    failed = sum(1 for _, _, s in results if s.startswith("❌"))
    for section, name, status in results:
        print(f"  {status}  [{section}] {name}")
    print("═" * 60)
    print(f"  TOTAL: {len(results)} | PASSED: {passed} | FAILED: {failed}")
    print("═" * 60)
    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
async def run_all(section_filter: str = ""):
    sections = {
        "connectivity": test_gemini_connectivity,
        "vision":       test_vision,
        "rag_build":    test_rag_build,
        "injection":    test_content_injection,
        "query":        test_query,
        "kg_inspector": test_kg_inspector,
    }

    if section_filter:
        if section_filter not in sections:
            print(f"Unknown section: {section_filter}. Options: {list(sections)}")
            sys.exit(1)
        await sections[section_filter]()
    else:
        for fn in sections.values():
            await fn()

    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--section", default="", help="Run only one test section")
    args = p.parse_args()
    asyncio.run(run_all(args.section))
