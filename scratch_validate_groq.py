"""
scratch_validate_groq.py
========================
Validates the full RAG pipeline end to end:
  1. Ingests CNN sample chunks into ChromaDB
  2. Queries "What are the main components of a CNN?"
  3. Shows exactly what was retrieved so you can confirm
     Groq is answering FROM the corpus, not from its own training data.

Validation method: compare the retrieved chunk text to the LLM answer.
If the answer references facts that are NOT in the retrieved chunks,
the hallucination guard is not working correctly.

Usage:
    uv run python scratch_validate_groq.py
"""

from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import AgentGraphBuilder
from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.vectorstore.store import VectorStoreManager

# ---------------------------------------------------------------------------
# Sample CNN chunks to ingest
# ---------------------------------------------------------------------------

CNN_CHUNKS = [
    {
        "source": "cnn_intermediate.md",
        "topic": "CNN",
        "difficulty": "intermediate",
        "type": "architecture",
        "related_topics": ["convolution", "pooling", "feature_maps"],
        "is_bonus": False,
        "text": (
            "A Convolutional Neural Network is composed of three main types of layers: "
            "convolutional layers, pooling layers, and fully connected layers. "
            "The convolutional layer applies a set of learnable filters across the input, "
            "producing feature maps that detect patterns such as edges, textures, and shapes. "
            "The pooling layer reduces the spatial dimensions of each feature map, "
            "retaining the most important information while reducing computation. "
            "The fully connected layer at the end of the network combines all learned features "
            "to produce the final classification output. This hierarchical structure allows "
            "CNNs to learn increasingly abstract representations from raw pixel data."
        ),
    },
    {
        "source": "cnn_beginner.md",
        "topic": "CNN",
        "difficulty": "beginner",
        "type": "concept_explanation",
        "related_topics": ["convolution", "stride", "padding"],
        "is_bonus": False,
        "text": (
            "The convolution operation is the core building block of a convolutional network. "
            "A small matrix called a filter or kernel slides across the input image with a "
            "fixed step size called the stride. At each position, the filter computes a "
            "dot product with the corresponding region of the input, producing a single value "
            "in the output feature map. Padding can be added around the input border to "
            "control the output size. Multiple filters are applied in parallel, each learning "
            "to detect a different visual pattern. The weights of each filter are learned "
            "during backpropagation."
        ),
    },
    {
        "source": "cnn_intermediate_pooling.md",
        "topic": "CNN",
        "difficulty": "intermediate",
        "type": "concept_explanation",
        "related_topics": ["max_pooling", "average_pooling", "translation_invariance"],
        "is_bonus": False,
        "text": (
            "Pooling layers reduce the spatial size of feature maps after convolution, "
            "lowering the number of parameters and providing a form of translation invariance. "
            "Max pooling takes the largest value in each pooling window, preserving the "
            "most prominent activation in that region. Average pooling computes the mean "
            "value instead. A common configuration is a 2x2 pooling window with stride 2, "
            "which halves the height and width of the feature map. Pooling makes the network "
            "less sensitive to the exact position of a feature in the input, which improves "
            "generalisation on unseen images."
        ),
    },
]


def build_chunks() -> list[DocumentChunk]:
    chunks = []
    for c in CNN_CHUNKS:
        metadata = ChunkMetadata(
            topic=c["topic"],
            difficulty=c["difficulty"],
            type=c["type"],
            source=c["source"],
            related_topics=c["related_topics"],
            is_bonus=c["is_bonus"],
        )
        chunks.append(
            DocumentChunk(
                chunk_id=VectorStoreManager.generate_chunk_id(c["source"], c["text"]),
                chunk_text=c["text"],
                metadata=metadata,
            )
        )
    return chunks


def main() -> None:
    # ---- Step 1: Ingest ------------------------------------------------
    print("=" * 60)
    print("STEP 1 — INGESTING CNN CHUNKS")
    print("=" * 60)

    manager = VectorStoreManager()
    chunks = build_chunks()
    result = manager.ingest(chunks)
    print(f"Ingested: {result.ingested}, Skipped: {result.skipped}, Errors: {len(result.errors)}")

    # ---- Step 2: Show what retrieval returns ---------------------------
    query = "What are the main components of a CNN?"
    print(f"\n{'=' * 60}")
    print("STEP 2 — RETRIEVED CHUNKS")
    print(f"Query: '{query}'")
    print("=" * 60)

    retrieved = manager.query(query, k=4)
    if not retrieved:
        print("No chunks retrieved — check similarity threshold or re-ingest.")
        return

    for i, chunk in enumerate(retrieved, 1):
        print(f"\n[Chunk {i}] Score: {chunk.score:.4f} | {chunk.to_citation()}")
        print(chunk.chunk_text)

    # ---- Step 3: Run through full LangGraph pipeline -------------------
    print(f"\n{'=' * 60}")
    print("STEP 3 — GROQ LLM RESPONSE")
    print("=" * 60)

    graph = AgentGraphBuilder().build()
    config = {"configurable": {"thread_id": "validate-cnn-001"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    response = result.get("final_response")
    if response is None:
        print("No response generated.")
        return

    print(f"\nRewritten query: {response.rewritten_query}")
    print(f"Confidence:      {response.confidence:.4f}")
    print(f"No context:      {response.no_context_found}")
    print(f"\nSources cited:")
    for s in response.sources:
        print(f"  {s}")
    print(f"\nAnswer:\n{response.answer}")

    # ---- Step 4: Validation guide --------------------------------------
    print(f"\n{'=' * 60}")
    print("VALIDATION CHECK")
    print("=" * 60)
    print("Compare the answer above to the retrieved chunks.")
    print("The answer should ONLY reference facts present in the chunks.")
    print("If the answer introduces facts not in the chunks above,")
    print("the system prompt constraints may need tightening.")


if __name__ == "__main__":
    main()
