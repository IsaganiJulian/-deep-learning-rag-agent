"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

  def chunk_file(
        self, file_path: Path, metadata_overrides: dict | None = None
    ) -> list[DocumentChunk]:
        """Loads and chunks a single file."""
        stem = file_path.stem
        parts = stem.split("_")
        
        # 1. Standardize Metadata
        topic = parts[0].upper() if parts else "GENERAL"
        difficulty = parts[1] if len(parts) > 1 else "intermediate"
        is_bonus = topic.lower() in {"som", "boltzmannmachine", "gan"}

        base_metadata = ChunkMetadata(
            topic=topic,
            difficulty=difficulty,
            type="concept_explanation",
            source=file_path.name,
            related_topics=[],
            is_bonus=is_bonus,
        )

        if metadata_overrides:
            for key, value in metadata_overrides.items():
                if hasattr(base_metadata, key):
                    setattr(base_metadata, key, value)

        # 2. Load Content
        suffix = file_path.suffix.lower()
        raw_documents = []
        
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            raw_documents = loader.load()
        elif suffix in (".md", ".markdown"):
            # Use Markdown splitting logic here
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_documents = splitter.split_text(f.read())
        
        # 3. Create Chunks with IDs
        chunks = []
        for doc in raw_documents:
            chunk_id = VectorStoreManager.generate_chunk_id(file_path.name, doc.page_content)
            
            # Copy base metadata and add page info if available
            meta = base_metadata.model_copy()
            if "page" in doc.metadata:
                meta.page_number = doc.metadata["page"]

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=doc.page_content,
                    metadata=meta
                )
            )
            
        return chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs = splitter.split_documents(pages)

        return [
            {"text": doc.page_content.strip(), "page": doc.metadata.get("page", 0) + 1}
            for doc in docs
            if doc.page_content.strip()
        ]

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )
        header_docs = header_splitter.split_text(content)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        split_docs = splitter.split_documents(header_docs)

        chunks: list[dict] = []
        for doc in split_docs:
            text = doc.page_content.strip()
            if not text:
                continue
            header = doc.metadata.get("h3") or doc.metadata.get("h2") or doc.metadata.get("h1")
            chunks.append(
                {
                    "text": text,
                    "header": header,
                }
            )
        return chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        stem = file_path.stem
        parts = stem.split("_")
        topic = parts[0] if parts else "general"
        difficulty = parts[1] if len(parts) > 1 else "intermediate"

        metadata_dict = {
            "topic": topic,
            "difficulty": difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": [],
            "is_bonus": topic.lower() in {"som", "boltzmannmachine", "gan"},
        }

        if overrides:
            for key, value in overrides.items():
                if key in metadata_dict:
                    metadata_dict[key] = value

        return ChunkMetadata(**metadata_dict)
