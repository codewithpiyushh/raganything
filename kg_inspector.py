"""
kg_inspector.py
───────────────
Inspect, visualise, and export the LightRAG / RAGAnything knowledge graph.

Run:
    python kg_inspector.py --working_dir ./rag_storage
"""

import asyncio
import json
import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict

log = logging.getLogger("kg_inspector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
#  Load graph files written by LightRAG
# ─────────────────────────────────────────────────────────────────────────────
def load_graph_data(working_dir: str) -> dict:
    """Load entity & relation JSON files from LightRAG storage."""
    data = {"entities": [], "relations": [], "chunks": []}

    files = {
        "entities":  "vdb_entities.json",
        "relations": "vdb_relationships.json",
        "chunks":    "vdb_chunks.json",
    }

    for key, fname in files.items():
        path = os.path.join(working_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            # vdb files store {id: {data: {...}}}
            if isinstance(raw, dict):
                data[key] = [v.get("data", v) for v in raw.values()]
            elif isinstance(raw, list):
                data[key] = raw
            log.info(f"Loaded {len(data[key])} {key}")
        else:
            log.warning(f"File not found: {path}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────
def print_statistics(data: dict):
    entities  = data["entities"]
    relations = data["relations"]
    chunks    = data["chunks"]

    # Entity type breakdown
    type_counts: dict[str, int] = defaultdict(int)
    for e in entities:
        etype = e.get("entity_type", "UNKNOWN")
        type_counts[etype] += 1

    # Relation keyword breakdown (top-10)
    kw_counts: dict[str, int] = defaultdict(int)
    for r in relations:
        for kw in r.get("keywords", "").split(","):
            kw = kw.strip()
            if kw:
                kw_counts[kw] += 1
    top_kws = sorted(kw_counts.items(), key=lambda x: -x[1])[:10]

    print("\n" + "═" * 55)
    print("  KNOWLEDGE GRAPH STATISTICS")
    print("═" * 55)
    print(f"  Total entities  : {len(entities)}")
    print(f"  Total relations : {len(relations)}")
    print(f"  Total chunks    : {len(chunks)}")
    print("\n  Entity type distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 40)
        print(f"    {etype:<20} {count:>4}  {bar}")
    print("\n  Top-10 relation keywords:")
    for kw, count in top_kws:
        print(f"    {kw:<25} {count:>4}")
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Sample display
# ─────────────────────────────────────────────────────────────────────────────
def sample_entities(data: dict, n: int = 5):
    print(f"\n── Sample entities (first {n}) ─────────────────────────────────")
    for e in data["entities"][:n]:
        name  = e.get("entity_name", e.get("id", "?"))
        etype = e.get("entity_type", "?")
        desc  = e.get("description", "")[:120]
        print(f"  [{etype}] {name}\n       {desc}")


def sample_relations(data: dict, n: int = 5):
    print(f"\n── Sample relations (first {n}) ─────────────────────────────────")
    for r in data["relations"][:n]:
        src  = r.get("src_id", "?")
        tgt  = r.get("tgt_id", "?")
        desc = r.get("description", "")[:100]
        kws  = r.get("keywords", "")
        wt   = r.get("weight", 0)
        print(f"  {src}  →  {tgt}  (w={wt:.2f})")
        print(f"    keywords: {kws}")
        print(f"    desc: {desc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Export to GraphML (for Gephi, Cytoscape, etc.)
# ─────────────────────────────────────────────────────────────────────────────
def export_graphml(data: dict, output_path: str = "./rag_knowledge_graph.graphml"):
    """Write a GraphML file from the loaded entities and relations."""
    try:
        import networkx as nx
    except ImportError:
        log.error("networkx not installed — run: pip install networkx")
        return

    G = nx.DiGraph()

    for e in data["entities"]:
        nid   = e.get("entity_name", e.get("id", "unknown"))
        etype = e.get("entity_type", "UNKNOWN")
        desc  = e.get("description", "")[:200]
        G.add_node(nid, entity_type=etype, description=desc)

    for r in data["relations"]:
        src  = r.get("src_id")
        tgt  = r.get("tgt_id")
        if src and tgt:
            desc = r.get("description", "")[:200]
            kws  = r.get("keywords", "")
            wt   = float(r.get("weight", 1.0))
            G.add_edge(src, tgt, description=desc, keywords=kws, weight=wt)

    nx.write_graphml(G, output_path)
    log.info(f"✅  GraphML exported → {output_path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  Export to JSON summary
# ─────────────────────────────────────────────────────────────────────────────
def export_summary_json(data: dict, output_path: str = "./kg_summary.json"):
    summary = {
        "entity_count":   len(data["entities"]),
        "relation_count": len(data["relations"]),
        "chunk_count":    len(data["chunks"]),
        "entities":  [
            {k: v for k, v in e.items() if k in ("entity_name", "entity_type", "description")}
            for e in data["entities"][:50]     # first 50 for brevity
        ],
        "relations": [
            {k: v for k, v in r.items() if k in ("src_id", "tgt_id", "keywords", "weight")}
            for r in data["relations"][:50]
        ],
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"✅  Summary JSON exported → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="RAGAnything Knowledge Graph Inspector")
    p.add_argument("--working_dir",  default="./rag_storage", help="RAGAnything storage dir")
    p.add_argument("--export_graphml", action="store_true", help="Export to GraphML")
    p.add_argument("--export_json",    action="store_true", help="Export summary JSON")
    p.add_argument("--sample_n",       type=int, default=5,  help="Entities/rels to preview")
    args = p.parse_args()

    data = load_graph_data(args.working_dir)

    if not any(data.values()):
        print("⚠️  No graph data found. Run rag_gemini_main.py --mode demo first.")
        return

    print_statistics(data)
    sample_entities(data,  n=args.sample_n)
    sample_relations(data, n=args.sample_n)

    if args.export_graphml:
        export_graphml(data)

    if args.export_json:
        export_summary_json(data)


if __name__ == "__main__":
    main()
