# System Architecture
## Team: Group 12
## Date: 03/23/2026
## Members and Roles:
- Corpus Architect: Kusuma
- Pipeline Engineer: Isagani
- UX Lead: Kusuma
- Prompt Engineer: Srijitha
- QA Lead: Hemanth

---

## Architecture Diagram

The diagram must show:
- [x] How a corpus file becomes a chunk
- [x] How a chunk becomes an embedding
- [x] How duplicate detection fires
- [x] How a user query flows through LangGraph to a response
- [x] Where the hallucination guard sits in the graph
- [x] How conversation memory is maintained across turns

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                             │
│                                                                         │
│  data/corpus/          DocumentChunker           VectorStoreManager     │
│  ┌──────────┐          ┌─────────────────┐       ┌──────────────────┐  │
│  │ .md file │─────────▶│ MarkdownHeader  │       │                  │  │
│  └──────────┘          │ TextSplitter    │       │  generate_       │  │
│                        │      +          │──────▶│  chunk_id()      │  │
│  ┌──────────┐          │ Recursive       │       │  SHA-256(src +   │  │
│  │ .pdf file│─────────▶│ CharSplitter    │       │  text)[:16]      │  │
│  └──────────┘          │                 │       │       │          │  │
│                        │ chunk_size=512  │       │       ▼          │  │
│                        │ overlap=50      │       │  check_duplicate │  │
│                        └─────────────────┘       │       │          │  │
│                               │                  │  dup? │ new?     │  │
│                               │ DocumentChunk    │  skip ▼ embed    │  │
│                               │ + ChunkMetadata  │  HuggingFace     │  │
│                               │                  │  all-MiniLM-L6   │  │
│                               └─────────────────▶│       │          │  │
│                                                   │  collection      │  │
│                                                   │  .upsert()       │  │
│                                                   │  ChromaDB        │  │
│                                                   │  PersistentClient│  │
│                                                   └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        QUERY / RESPONSE PIPELINE                        │
│                                                                         │
│  User query                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    LangGraph Agent (AgentState)                   │  │
│  │                                                                   │  │
│  │  ┌──────────────────┐    ┌──────────────────┐                    │  │
│  │  │ query_rewrite_   │    │  retrieval_node  │                    │  │
│  │  │ node             │───▶│                  │                    │  │
│  │  │                  │    │  embed query     │                    │  │
│  │  │ QUERY_REWRITE_   │    │  cosine search   │                    │  │
│  │  │ PROMPT           │    │  threshold=0.3   │                    │  │
│  │  │                  │    │  k=4 chunks      │                    │  │
│  │  │ expands to       │    │       │          │                    │  │
│  │  │ keyword-dense    │    │  no chunks? ─────┼──▶ no_context_     │  │
│  │  │ technical query  │    │  set flag        │    found = True    │  │
│  │  └──────────────────┘    └──────────────────┘         │         │  │
│  │                                   │                    │         │  │
│  │                           chunks found?                │         │  │
│  │                                   │                    ▼         │  │
│  │                                   ▼          ┌──────────────┐   │  │
│  │                          ┌──────────────────┐│  END (guard) │   │  │
│  │                          │ generation_node  ││              │   │  │
│  │                          │                  ││ "No relevant │   │  │
│  │                          │ SYSTEM_PROMPT +  ││  context     │   │  │
│  │                          │ retrieved chunks ││  found…"     │   │  │
│  │                          │ + message history││              │   │  │
│  │                          │                  │└──────────────┘   │  │
│  │                          │ trim_messages()  │                   │  │
│  │                          │ if > 3000 tokens │                   │  │
│  │                          │       │          │                   │  │
│  │                          │  Groq LLM call   │                   │  │
│  │                          │  llama-3.1-8b    │                   │  │
│  │                          └──────────────────┘                   │  │
│  │                                   │                              │  │
│  │  MemorySaver checkpointer ◀────── AgentResponse                 │  │
│  │  (keyed by thread_id)             answer + sources + confidence  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                   │                                     │
│                                   ▼                                     │
│                           Streamlit UI                                  │
│                           chat bubble + citation expanders              │
└─────────────────────────────────────────────────────────────────────────┘
```


---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:** Both `.md` (Markdown) and `.pdf` (PDF). Markdown files use
  header-aware splitting; PDFs use PyPDFLoader followed by recursive character splitting.

- **Landmark papers ingested:**
  - Rumelhart, Hinton & Williams (1986) — Backpropagation
  - LeCun et al. (1998) — LeNet / CNNs
  - Hochreiter & Schmidhuber (1997) — LSTM
  - Sutskever, Vinyals & Le (2014) — Seq2Seq
  - Hinton & Salakhutdinov (2006) — Autoencoders

- **Chunking strategy:**
  512-character chunks with 50-character overlap, applied after format-specific splitting.

  For **Markdown**, `MarkdownHeaderTextSplitter` first splits on `#`, `##`, and `###`
  headers to keep semantically coherent sections together, then `RecursiveCharacterTextSplitter`
  handles any sections that still exceed 512 characters. This preserves semantic coherence —
  a concept within one section stays within one chunk rather than being split across boundaries.

  For **PDF**, `PyPDFLoader` extracts text page-by-page, then `RecursiveCharacterTextSplitter`
  applies the same 512/50 parameters. Page numbers are preserved in metadata.

  512 characters balances context richness (enough surrounding text for the LLM to reason from)
  with retrieval precision (smaller chunks score higher against specific queries). The 50-character
  overlap prevents concepts that span chunk boundaries from being lost — a standard production
  technique for avoiding "split sentence" retrieval failures.

- **Metadata schema:**
  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | Primary deep learning topic (ANN, CNN, LSTM, etc.) — enables topic-scoped retrieval via ChromaDB `where` filter |
  | difficulty | string | One of beginner / intermediate / advanced — inferred from filename convention (`lstm_intermediate.md`); drives LLM depth calibration |
  | type | string | Content category (concept_explanation, architecture, training_process, use_case, comparison, mathematical_foundation) — allows future filtering by content kind |
  | source | string | Filename of the originating document — used for citations displayed to the user and in deduplication |
  | related_topics | list (stored as comma-joined string) | Conceptually linked topics — available for context enrichment and graph-style multi-hop retrieval |
  | is_bonus | bool (stored as "true"/"false") | Flags SOM, BoltzmannMachine, and GAN chunks so the UI can surface bonus material separately |
  | page_number | int (–1 if absent) | PDF page number where the chunk originated — included in citation strings for traceability |

- **Duplicate detection approach:**
  Each chunk receives a deterministic 16-character ID generated as the first 16 hex digits of
  `SHA-256(source_filename + "::" + chunk_text)`. Before any embed call, `check_duplicate()`
  queries ChromaDB for that ID; if it exists, the chunk is skipped and the `skipped` counter
  increments.

  A content hash is more reliable than a filename-based ID because filenames change (renaming,
  re-uploading, versioning) while content stays the same. Two uploads of `lstm_v2.md` with
  identical text to `lstm.md` will produce the same chunk IDs and be skipped correctly —
  something filename-based deduplication would miss entirely.

- **Corpus coverage:**
  - [x] ANN — 5 chunks (forward propagation, backprop, activation functions, loss functions, vanishing gradients)
  - [x] CNN — 5 chunks (convolution operation, pooling, feature maps, LeNet, AlexNet)
  - [x] RNN — 4 chunks (hidden state, BPTT, vanishing gradients, applications)
  - [x] LSTM — 5 chunks (vanishing gradient solution, forget gate, input gate, output gate, LSTM vs RNN)
  - [x] Seq2Seq — 5 chunks (encoder-decoder, encoder, decoder, relation to autoencoders, attention)
  - [x] Autoencoder — 5 chunks (architecture, latent space, denoising AE, variational AE, applications)
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [ ] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** ChromaDB — PersistentClient
- **Local persistence path:** `./data/chroma_db`

- **Embedding model:**
  all-MiniLM-L6-v2 via sentence-transformers (HuggingFaceEmbeddings), running locally on CPU

- **Why this embedding model:**
  Chosen for speed and zero API cost. all-MiniLM-L6-v2 is lightweight (~90MB) and loads from local cache after the first run. No data leaves the machine, which matters for proprietary corpus content. The tradeoff is slightly lower quality than OpenAI embeddings, which is acceptable for a study corpus of this size.

- **Similarity metric:**
  Cosine similarity. ChromaDB collection is configured with `{"hnsw:space": "cosine"}`. Cosine measures the angle between vectors rather than Euclidean distance, making it robust to differences in chunk length — a short chunk and a long chunk on the same topic will still score highly if their content is semantically aligned.

- **Retrieval k:**
  4 chunks per query (RETRIEVAL_K=4). Enough to provide multi-angle context without overloading the LLM's context window. Increasing k beyond 6–8 starts to introduce noise and low-relevance chunks.

- **Similarity threshold:**
  0.3 (SIMILARITY_THRESHOLD=0.3). Calibrated manually by running test queries and printing raw scores. Scores below 0.3 were consistently unrelated to the query topic. Scores above 0.3 were reliably on-topic. The hallucination guard fires when no chunk meets this threshold.

- **Metadata filtering:**
  Users can filter retrieval by topic (e.g. "LSTM") and difficulty (e.g. "intermediate"). Filters are passed as ChromaDB `where` clauses in `VectorStoreManager.query()`. If no filter is provided, retrieval runs across the full corpus.

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**
  | Node | Responsibility |
  |---|---|
  | query_rewrite_node | Rewrites the user's natural language question into a keyword-dense search query optimised for vector similarity retrieval |
  | retrieval_node | Queries ChromaDB with the rewritten query and sets the no_context_found flag if no chunks meet the similarity threshold |
  | generation_node | Builds the prompt from retrieved chunks and conversation history, calls the LLM, and returns a structured AgentResponse with citations |

- **Conditional edges:**
  After retrieval_node, `should_retry_retrieval` checks the `no_context_found` flag. If True, the graph routes to END and skips generation — the hallucination guard fires. If False, the graph routes to generation_node to produce an answer.

- **Hallucination guard:**
  When no chunks meet the similarity threshold, the system returns:
  > "I was unable to find relevant information in the corpus for your query. This may mean the topic is not yet covered in the study material, or your query may need to be rephrased. Please try a more specific deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."

- **Query rewriting:**
  - Raw query: "What are the main components of a CNN?"
  - Rewritten query: "convolutional neural network architecture, convolutional layers, pooling layers, activation functions, weights, biases, feature extraction"

- **Conversation memory:**
  LangGraph's MemorySaver checkpointer maintains full message history per thread_id. Each user session has a unique thread_id so histories do not bleed between users. When history approaches MAX_CONTEXT_TOKENS (3000), `trim_messages` with strategy="last" removes the oldest non-system messages, preserving the most recent context.

- **LLM provider:**
  Groq — llama-3.1-8b-instant

- **Why this provider:**
  Groq's LPU (Language Processing Unit) inference delivers significantly lower latency than GPU-based providers. The free tier is sufficient for a class project, and llama-3.1-8b-instant produces fast, accurate responses for deep learning Q&A without requiring a paid API plan.

---

### Prompt Layer

- **System prompt summary:**
  The agent is cast as a senior machine learning engineer running a technical interview
  preparation session. Five strict rules are enforced in the prompt:
  1. Answer only from the provided context — no general knowledge.
  2. If the context is insufficient, say so clearly rather than inferring.
  3. Cite every factual claim using `[SOURCE: topic | filename]` format.
  4. Adjust technical depth to match the difficulty level in the chunk metadata.
  5. When a student answer is partially correct, acknowledge what is right before explaining gaps.

  Tone instruction: *"Clear, technically precise, encouraging but rigorous — like a fair senior
  engineer who wants the candidate to succeed but will not lower the bar."*

- **Question generation prompt:**
  **Inputs:** `{context}` (retrieved chunk text), `{difficulty}` (from chunk metadata).
  **Returns:** A JSON object with six fields:
  - `question` — the interview question
  - `difficulty` — echoed back for validation
  - `topic` — primary concept being tested
  - `model_answer` — complete, accurate answer drawn strictly from the source material
  - `follow_up` — a probe question to test deeper understanding
  - `source_citations` — list of `[SOURCE: topic | filename]` strings

- **Answer evaluation prompt:**
  **Inputs:** `{question}`, `{candidate_answer}`, `{context}` (ground-truth source material).
  **Returns:** A JSON object scored 0–10 with six fields:
  - `score` — integer 0–10
  - `what_was_correct` — specific aspects the candidate got right
  - `what_was_missing` — absent or incorrect concepts
  - `ideal_answer` — model answer drawn strictly from source material
  - `interview_verdict` — one of: `hire`, `consider`, `no hire`
  - `coaching_tip` — one specific study recommendation

  **Scoring rubric:**
  | Score | Meaning |
  |---|---|
  | 9–10 | Complete, accurate, well-articulated — ready for senior roles |
  | 7–8 | Mostly correct with minor gaps — good junior to mid-level candidate |
  | 5–6 | Core concept understood but significant details missing |
  | 3–4 | Partial understanding, notable misconceptions present |
  | 0–2 | Fundamental misunderstanding or no relevant knowledge demonstrated |

- **JSON reliability:**
  Both the question generation and answer evaluation prompts include an exact JSON template
  with literal field names and types as a schema. Every prompt closes with:
  *"Respond with the JSON object only. No preamble or explanation."*
  This eliminates leading prose, markdown code fences, and trailing commentary that would
  break `json.loads()` parsing.

- **Failure modes identified:**
  - **System prompt:** LLM uses general knowledge despite the "context only" rule —
    addressed by placing the rule first and repeating "drawn strictly from the source material"
    in both the question generation and answer evaluation prompts.
  - **Query rewrite prompt:** LLM returns a conversational sentence instead of a keyword string —
    addressed by specifying "Maximum 15 words", "Output only the rewritten query, nothing else",
    and providing a before/after example in the architecture docs.
  - **Question generation prompt:** LLM generates a question that cannot be answered from the
    provided chunk (too broad) — addressed by instructing it to connect "at least two concepts
    from the source material" rather than introducing external concepts.
  - **Answer evaluation prompt:** LLM gives inflated scores to avoid seeming harsh —
    addressed by including the explicit 5-tier scoring rubric with outcome labels
    (hire / consider / no hire) that anchor the score to real-world consequences.

---

### Interface Layer

- **Framework:** Streamlit
- **Deployment platform:** *(Streamlit Community Cloud / HuggingFace Spaces)*
- **Public URL:** *(paste your deployed app URL here once live)*

- **Ingestion panel features:**
  Located in the sidebar. Users drag or browse to upload one or more `.pdf` or `.md` files.
  Selected files are listed with icons (📕 PDF / 📝 Markdown) before ingestion begins.
  An "Ingest files" button triggers chunking and embedding. A success banner reports the
  count of chunks ingested, duplicates skipped, and any errors. If no chunks are produced
  (e.g. an empty file), an explicit error is shown rather than silently succeeding.

- **Document viewer features:**
  Displayed in the left main column. A selectbox lets users switch between all ingested
  documents. The selected document's name, topic, difficulty, and chunk count are shown as
  metadata badges pulled live from ChromaDB. Below the badges, a scrollable 280px container
  renders each chunk's full text with a numbered label ("Chunk 1", "Chunk 2", …).
  On app load, the viewer is pre-populated from ChromaDB so existing corpus is visible
  immediately — no re-upload required after a restart.

- **Chat panel features:**
  Located in the right main column. Users type a question in a text input; the agent rewrites
  the query, retrieves chunks, and generates a response. The assistant's answer appears in a
  styled chat bubble. Each source chunk is shown in a collapsible expander titled
  "📎 Source N · referenced passage" displaying the topic, difficulty, source filename, and
  the raw chunk text. When the hallucination guard fires (no chunks above the 0.3 threshold),
  the response message explains why no answer was produced and suggests rephrasing with specific
  deep learning terminology. A "Clear chat" button resets history and generates a new thread_id.
  Suggested prompt chips are shown when the chat is empty to help users get started.

- **Session state keys:**
  | Key | Stores |
  |---|---|
  | chat_history | List of `{"role": str, "content": str, "sources": list}` dicts — the full conversation displayed in the chat panel |
  | chat_thread_id | UUID string identifying the current conversation — passed to MemorySaver so LangGraph maintains per-session history |
  | uploaded_names | List of source filenames currently known to ChromaDB — drives the document viewer selectbox |
  | selected_document | The document name currently shown in the viewer — bound to the selectbox widget |
  | show_reply_toast | Boolean flag that triggers a toast notification after a successful agent response |
  | prompt_queue | Holds a suggested prompt chip label so it can be submitted on the next rerun without double-firing |

- **Stretch features implemented:**
  Animated hero header with gradient blobs. Quick-start prompt chips. Per-source citation expanders with metadata tags. Warm/cool visual design system with light/dark mode support via CSS variables.

---

## Design Decisions

Document at least three deliberate decisions your team made.
These are your Hour 3 interview talking points — be specific.
"We used the default settings" is not a design decision.

1. **Decision:** Content-hash chunk IDs using SHA-256 of (source + chunk_text)
   **Rationale:** Filename-based IDs fail when files are renamed or re-uploaded. A content hash produces the same ID for identical content regardless of filename, making duplicate detection reliable across multiple ingest runs.
   **Interview answer:** We generate chunk IDs from a SHA-256 hash of the source filename and chunk text. This means two uploads of the same file always produce the same IDs, so our deduplication logic fires correctly even if the file is renamed or re-uploaded — something filename-based IDs would miss entirely.

2. **Decision:** Cosine similarity with a 0.3 threshold for the hallucination guard
   **Rationale:** Cosine similarity is invariant to chunk length, making it a fairer metric than Euclidean distance across chunks of varying size. The 0.3 threshold was calibrated by printing raw similarity scores during testing — scores below 0.3 were consistently off-topic.
   **Interview answer:** We use cosine similarity because it measures semantic alignment regardless of how long a chunk is, which matters when chunks range from 100 to 300 words. The 0.3 threshold was set empirically by running test queries and observing that below that score the retrieved chunks were no longer topically relevant.

3. **Decision:** Query rewriting node before retrieval
   **Rationale:** Users phrase questions conversationally, but documents are written technically. Rewriting the query into keyword-dense technical language before embedding significantly improves retrieval recall. Without this step, "I'm confused about how LSTMs remember things" would retrieve poorly compared to "LSTM cell state forget gate long-term memory mechanism".
   **Interview answer:** We added a query rewrite step because natural language questions and technical documents live in different parts of the embedding space. By rewriting the query into keyword-dense technical terminology before retrieval, we close that gap and consistently retrieve more relevant chunks.

4. **Decision:** MemorySaver checkpointer with per-session thread_id for conversation memory
   **Rationale:** Without a checkpointer, each graph invocation is stateless and the agent has no memory of prior turns. MemorySaver persists the full message history in memory keyed by thread_id, enabling multi-turn conversations without a database. History is trimmed with `trim_messages` when it approaches MAX_CONTEXT_TOKENS to prevent context window overflow.
   **Interview answer:** We use LangGraph's MemorySaver with a unique thread_id per user session so each conversation maintains its own history without interfering with others. When the history grows too long we trim the oldest messages first, always preserving the system prompt and most recent context.

---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query — "Explain the vanishing gradient problem" | Relevant chunks retrieved, accurate answer, source cited | Chunks from ANN and RNN files retrieved, answer cites `[ANN \| ann_intermediate.md]` | Pass |
| Off-topic query — "What is the capital of France" | Hallucination guard fires, clear no-context message | Guard message returned, no fabricated deep learning content | Pass |
| Duplicate ingestion — upload same file twice | Second upload detected and skipped | `IngestionResult.skipped` equals chunk count on second run | Pass |
| Empty query — submit blank input | Graceful error, no crash | Streamlit chat_input does not submit empty strings; no crash observed | Pass |
| Cross-topic query — "How do LSTMs improve on RNNs for Seq2Seq" | Chunks from at least two topics retrieved | Chunks from `lstm_intermediate.md` and `rnn_intermediate.md` both returned | Pass |

**Critical failures fixed before Hour 3:**
- `chunker.py` had an indentation error on `chunk_file` — fixed and `chunk_files()` method added.
- `should_retry_retrieval` always returned `"generate"` — generation_node now handles the `no_context_found` branch internally with the guard message.

**Known issues not fixed (and why):**
- Conversation memory does not persist across app restarts (MemorySaver is in-process only — acceptable for a demo, would require a persistent database in production).
- PDF chunking may produce noisy chunks from reference sections in academic papers — not filtered in the current pipeline due to time constraints.

---

## Known Limitations

- **PDF chunking noise:** `PyPDFLoader` extracts all text including headers, footers, page
  numbers, and reference lists. These noisy chunks can surface during retrieval and dilute
  answer quality. Markdown files do not have this problem because the structure is explicit.

- **Manually calibrated similarity threshold:** The 0.3 threshold was set by inspecting raw
  scores during manual testing — not via a labelled evaluation set. It may be too permissive
  for some query types and too strict for others.

- **In-memory conversation history:** `MemorySaver` stores message history in RAM. All
  conversation context is lost when the Streamlit app restarts or the process is recycled.
  There is no persistent session database.

- **Metadata inferred from filename convention:** Topic and difficulty are parsed from the
  filename stem (e.g. `lstm_intermediate.md`). Files that do not follow this convention
  receive generic defaults (`topic="lstm"`, `difficulty="intermediate"`) and require manual
  overrides. Incorrectly named files silently produce wrong metadata.

- **No re-ranking step:** Retrieved chunks are ranked by cosine similarity alone. A
  cross-encoder re-ranker would produce more accurate relevance ordering, especially for
  multi-sentence queries where embedding similarity is a coarser signal.

- **Single-document viewer:** The document viewer shows one document at a time. There is no
  full-corpus search or side-by-side comparison of chunks from different topics.

---

## What We Would Do With More Time

- **Hybrid search (BM25 + vector):** Combine dense vector retrieval with sparse BM25 keyword
  search and fuse the results. Hybrid search consistently outperforms pure vector search on
  technical queries where exact term matching matters (e.g. "LSTM forget gate equation").

- **Cross-encoder re-ranking:** After retrieving the top-k chunks by cosine similarity, pass
  (query, chunk) pairs through a cross-encoder model that scores true relevance. This
  two-stage approach significantly reduces false positives in the retrieved context.

- **Async ingestion:** Move chunking and embedding into a background task so large PDFs do not
  block the Streamlit UI thread. Display a progress bar rather than a blocking spinner.

- **Persistent conversation memory:** Replace `MemorySaver` with a database-backed
  checkpointer (SQLite or Redis) so conversation history survives app restarts and can be
  resumed across sessions.

- **PDF noise filtering:** Add a post-processing step after `PyPDFLoader` to strip reference
  sections, running headers/footers, and equation-only lines before chunking.

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:** Why does your system use cosine similarity rather than Euclidean distance
for chunk retrieval, and what does your similarity threshold represent?

Model answer: Cosine similarity measures the angle between two embedding vectors, making it
invariant to the magnitude (length) of the vectors. This means a short chunk and a long chunk
on the same topic will still score highly if their semantic content is aligned — something
Euclidean distance would penalise because longer vectors have larger magnitude. The similarity
threshold is the minimum cosine score a retrieved chunk must achieve to be included in the
LLM context. If no chunk meets it, the hallucination guard fires and the system declines to
answer rather than generating an unsupported response.

**Question 2:** What happens in your system when a user asks about a topic that is not in
the corpus? Walk through the full execution path.

Model answer: The user query enters `query_rewrite_node`, which rewrites it into a
keyword-dense technical query. `retrieval_node` embeds that query, runs a cosine similarity
search against ChromaDB, and applies the similarity threshold filter. If no chunk scores above
the threshold, `no_context_found` is set to `True` on the `AgentState`. The conditional edge
`should_retry_retrieval` detects this flag and routes the graph directly to `END`, bypassing
`generation_node` entirely. The UI receives the `NO_CONTEXT_RESPONSE` message explaining that
the topic is not covered and suggesting the user rephrase or check corpus coverage.

**Question 3:** How does your duplicate detection work, and why would filename-based
deduplication fail in a production environment?

Model answer: Each chunk is assigned a deterministic ID computed as the first 16 hex
characters of `SHA-256(source_filename + "::" + chunk_text)`. Before embedding, the system
calls `check_duplicate()` which queries ChromaDB for that ID. If it already exists, the chunk
is skipped. Filename-based deduplication fails because the same content can arrive under
different filenames — a renamed file, a re-upload, or a version suffix will all produce a
new filename-based ID and be treated as new content. A content hash detects identical text
regardless of what the file is called.

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
- The LangGraph graph structure made the agent's control flow explicit and easy to debug — adding print statements at each node immediately showed where state was or wasn't passing through.
- Header-aware Markdown splitting produced noticeably cleaner chunks than naive character splitting on the same files.
- The content-hash chunk ID approach made duplicate detection completely reliable with zero extra bookkeeping.

**What confused us:**
- `trim_messages` from `langchain_core.messages` has a non-obvious API — the `token_counter=len` argument counts messages rather than tokens, which required careful tuning of `MAX_CONTEXT_TOKENS`.
- ChromaDB `where` filters require the exact field name and value from the stored metadata, not the Python object attribute names — this caused silent retrieval failures until we matched field names exactly.

**One thing each team member would study before a real interview:**
- Corpus Architect (Kusuma): Chunk quality evaluation — how to measure retrieval precision and recall against a labeled test set rather than eyeballing results.
- Pipeline Engineer (Isagani): Advanced RAG patterns — hybrid search (BM25 + vector), re-ranking with cross-encoders, and parent-document retrieval strategies.
- UX Lead (Kusuma): Streamlit async patterns — how to use `st.fragment` and background threads to prevent the UI from blocking during long operations.
- Prompt Engineer (Srijitha): Structured output reliability — how to use function calling / tool use APIs to guarantee JSON schema compliance rather than relying on prompt instructions alone.
- QA Lead (Hemanth): LangGraph evaluation frameworks — how to use LangSmith or RAGAS to quantitatively score retrieval quality and answer faithfulness at scale.
