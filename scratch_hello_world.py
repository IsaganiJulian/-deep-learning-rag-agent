"""
scratch_hello_world.py
======================
Hello world test for Pipeline Engineer Phase 1 milestone.

Run twice to verify:
  - First run: chunk is ingested, query returns a result with a score
  - Second run: chunk is skipped (duplicate detection fires)

Usage:
    uv run python scratch_hello_world.py
"""

import json
from pathlib import Path

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.vectorstore.store import VectorStoreManager


def main() -> None:
    # Load sample chunk
    sample_path = Path("examples/sample_chunk.json")
    data = json.loads(sample_path.read_text())

    metadata = ChunkMetadata(
        topic=data["metadata"]["topic"],
        difficulty=data["metadata"]["difficulty"],
        type=data["metadata"]["type"],
        source=data["metadata"]["source"],
        related_topics=data["metadata"]["related_topics"],
        is_bonus=data["metadata"]["is_bonus"],
    )

    chunk = DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id(
            metadata.source, data["chunk_text"]
        ),
        chunk_text=data["chunk_text"],
        metadata=metadata,
    )

    print(f"Chunk ID: {chunk.chunk_id}")

    # Ingest
    manager = VectorStoreManager()
    result = manager.ingest([chunk])
    print(f"Ingested: {result.ingested}, Skipped: {result.skipped}, Errors: {len(result.errors)}")

    # Query
    query = "what is a neural network"
    print(f"\nQuerying: '{query}'")
    retrieved = manager.query(query, k=4)

    if retrieved:
        for i, r in enumerate(retrieved, 1):
            print(f"\n[{i}] Score: {r.score:.4f}")
            print(f"    Source: {r.metadata.source} | Topic: {r.metadata.topic}")
            print(f"    Text: {r.chunk_text[:120]}...")
    else:
        print("No chunks returned above similarity threshold.")


if __name__ == "__main__":
    main()
