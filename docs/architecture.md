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

Replace this section with your team's completed flow chart.
Export from FigJam, Miro, or draw.io and embed as an image,
or describe the architecture as an ASCII diagram.

The diagram must show:
- [ ] How a corpus file becomes a chunk
- [ ] How a chunk becomes an embedding
- [ ] How duplicate detection fires
- [ ] How a user query flows through LangGraph to a response
- [ ] Where the hallucination guard sits in the graph
- [ ] How conversation memory is maintained across turns

*(replace this line with your diagram image or ASCII art)*

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:**
  *(which file types did your team ingest — .md, .pdf, or both?)*

- **Landmark papers ingested:**
  *(list the papers your team located and ingested, one per line)*
  -
  -
  -

- **Chunking strategy:**
  *(what chunk size and overlap did you choose, and why?
  e.g. 512 characters with 50 overlap — justify this choice)*

- **Metadata schema:**
  *(list every metadata field your chunks carry and explain why each field exists)*
  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | |
  | difficulty | string | |
  | type | string | |
  | source | string | |
  | related_topics | list | |
  | is_bonus | bool | |

- **Duplicate detection approach:**
  *(how is the chunk ID generated? why is a content hash more reliable than a filename?)*

- **Corpus coverage:**
  - [ ] ANN
  - [ ] CNN
  - [ ] RNN
  - [ ] LSTM
  - [ ] Seq2Seq
  - [ ] Autoencoder
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
  *(describe the agent persona and the key constraints in your system prompt)*

- **Question generation prompt:**
  *(what inputs does it take and what does it return?)*

- **Answer evaluation prompt:**
  *(how does it score a candidate answer? what is the scoring rubric?)*

- **JSON reliability:**
  *(what did you add to your prompts to ensure consistent JSON output?)*

- **Failure modes identified:**
  *(list at least one failure mode per prompt and how you addressed it)*
  -
  -
  -

---

### Interface Layer

- **Framework:** *(Streamlit / Gradio)*
- **Deployment platform:** *(Streamlit Community Cloud / HuggingFace Spaces)*
- **Public URL:** *(paste your deployed app URL here once live)*

- **Ingestion panel features:**
  *(describe what the user sees — file uploader, status display, document list)*

- **Document viewer features:**
  *(describe how users browse ingested documents and chunks)*

- **Chat panel features:**
  *(describe how citations appear, how the hallucination guard is surfaced,
  and any filters available)*

- **Session state keys:**
  *(list the st.session_state keys your app uses and what each stores)*
  | Key | Stores |
  |---|---|
  | chat_history | |
  | ingested_documents | |
  | selected_document | |
  | thread_id | |

- **Stretch features implemented:**
  *(streaming responses, async ingestion, hybrid search, re-ranking, other)*

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
| Normal query | Relevant chunks, source cited | | |
| Off-topic query | No context found message | | |
| Duplicate ingestion | Second upload skipped | | |
| Empty query | Graceful error, no crash | | |
| Cross-topic query | Multi-topic retrieval | | |

**Critical failures fixed before Hour 3:**
-
-

**Known issues not fixed (and why):**
-
-

---

## Known Limitations

Be honest. Interviewers respect candidates who understand
the boundaries of their own system.

- *(e.g. PDF chunking produces noisy chunks from reference sections)*
- *(e.g. similarity threshold was calibrated manually, not empirically)*
- *(e.g. conversation memory is lost when the app restarts)*

---

## What We Would Do With More Time

- *(e.g. implement hybrid search combining vector and BM25 keyword search)*
- *(e.g. add a re-ranking step using a cross-encoder)*
- *(e.g. async ingestion so large PDFs don't block the UI)*

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:**

Model answer:

**Question 2:**

Model answer:

**Question 3:**

Model answer:

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
-

**What confused us:**
-

**One thing each team member would study before a real interview:**
- Corpus Architect:
- Pipeline Engineer:
- UX Lead:
- Prompt Engineer:
- QA Lead:
