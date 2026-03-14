# Hour 2 — Hackathon Build Sprint
## Graduate Data Science Capstone — Deep Learning RAG Agent

---

## Overview

Hour 2 is a structured 90-minute build sprint organized into two phases separated
by a mandatory team standup. F2F students work with the instructor circulating.
Online students treat this as a project kickoff — use class time to establish
momentum and continue async over the following week.

**The deliverable:** A working RAG-powered interview preparation agent that ingests
deep learning study material, stores it in ChromaDB, and allows a user to chat with
that knowledge base to generate and answer technical interview questions.

---

## Phase 1 — Parallel Build (0:00 – 0:40)

Everyone starts simultaneously in their role. No waiting for others to finish first.
The goal of Phase 1 is to reach your individual milestone so Phase 2 integration
is not blocked.

---

### Corpus Architect — Phase 1 Milestone
**Goal: First three topics chunked, validated, and ready to ingest**

#### Step 1 — Set Up Your Document Structure (5 min)
Decide on your file format now and do not change it mid-sprint. Recommended:
Markdown (`.md`) files, one per topic. This makes chunking predictable and
version control clean. Name files clearly: `ann.md`, `cnn.md`, `rnn.md`, etc.

Place all corpus files in the `data/corpus/` directory of the starter repo.

#### Step 2 — Confirm Your Chunk Schema with the Pipeline Engineer (5 min)
Before writing a single chunk, sit with (or message) your Pipeline Engineer and
confirm the metadata fields. Every chunk must follow this schema or ingestion
will fail:

```json
{
  "chunk_text": "...",
  "metadata": {
    "topic": "LSTM",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "lstm.md",
    "related_topics": ["RNN", "vanishing_gradient", "Seq2Seq"],
    "is_bonus": false
  }
}
```

**Bonus topics** (SOM, Boltzmann Machines, GANs) must have `"is_bonus": true`
in their metadata so the UI can surface them appropriately.

#### Step 3 — Draft Content Using AI + Primary Sources (25 min)
Use AI aggressively to draft content, then edit for accuracy and depth.
You are the expert reviewer, not the transcriptionist.

**Required: At least one landmark paper per core topic.**
Locate the PDF, ingest it into your RAG system, and supplement with your own
authored chunks that synthesize the key ideas. The combination of a primary
source PDF and your authored study notes is what makes your corpus defensible
in Hour 3.

Landmark papers by topic:

| Topic | Paper | Where to Find |
|---|---|---|
| ANN / Backprop | Rumelhart, Hinton & Williams (1986) | Google Scholar |
| CNN | LeCun et al. (1998) LeNet | Yann LeCun's site |
| CNN (deep) | Krizhevsky et al. (2012) AlexNet | papers.nips.cc |
| RNN | Elman (1990) | Google Scholar |
| LSTM | Hochreiter & Schmidhuber (1997) | IDSIA.ch |
| Seq2Seq | Sutskever, Vinyals & Le (2014) | arxiv.org |
| Autoencoder | Hinton & Salakhutdinov (2006) | Science journal |
| GAN *(bonus)* | Goodfellow et al. (2014) | arxiv.org |
| Boltzmann *(bonus)* | Hinton & Sejnowski (1986) | Google Scholar |
| SOM *(bonus)* | Kohonen (1982) | Google Scholar |

**Chunk writing rules:**
- One atomic idea per chunk — if it could answer five interview questions, split it
- Minimum 100 words per chunk, maximum 300 words
- If you remove the topic name from the chunk, the content should still identify
  the topic clearly
- Each chunk should be able to stand alone as the basis for one interview question

#### Step 4 — Self-Review Before Standup (5 min)
Before the standup, verify:
- [ ] At least 3 topics drafted with a minimum of 3 chunks each
- [ ] All chunks follow the agreed metadata schema
- [ ] At least one landmark paper PDF located and downloaded
- [ ] No topic bleeding — each chunk covers exactly one concept

**Red flag:** If you are still deciding what topics to cover at the 40-minute
mark, you are behind. Flag this immediately during standup.

---

### Pipeline Engineer — Phase 1 Milestone
**Goal: ChromaDB initialised, VectorStoreManager stub wired, retrieval returning results**

#### Step 1 — Environment Setup (5 min)
```bash
# Clone starter repo and set up environment
git clone <team-repo-url>
cd deep-learning-rag-agent
uv sync
cp .env.example .env
# Add your LLM provider key to .env - see README for provider options
```

Verify your environment is working:
```bash
uv run python -c "import chromadb; import langchain; import langgraph; print('OK')"
```

If this fails, do not proceed. Fix dependencies first.

#### Step 2 — Implement VectorStoreManager (15 min)
Implement the `VectorStoreManager` class in `src/rag_agent/vectorstore/store.py`.
Start with these three methods in this order:

1. `_get_or_create_collection()` — initialize ChromaDB persistent client and
   collection. Use `chromadb.PersistentClient(path=settings.chroma_db_path)`.

2. `check_duplicate(doc_id: str) -> bool` — query the collection for an existing
   document with this ID. This is your guard against repeat ingestion.

3. `ingest(chunks: list[dict]) -> IngestionResult` — iterate chunks, check
   duplicate for each, embed and upsert non-duplicates. Return a result object
   with counts of ingested vs skipped.

Do not implement `query()` yet — that comes in Phase 2.

#### Step 3 — Agree API Contract with UX Lead (5 min)
Write down and share these exact method signatures. The UX Lead will code to these:

```python
# Ingestion endpoint
def ingest(self, chunks: list[dict]) -> IngestionResult:
    # Returns: IngestionResult(ingested=int, skipped=int, errors=list[str])

# Query endpoint  
def query(self, query_text: str, k: int = 4, filters: dict | None = None
          ) -> list[RetrievedChunk]:
    # Returns: list of RetrievedChunk(text=str, metadata=dict, score=float)

# Chat endpoint (agent)
def chat(self, query: str, history: list[dict]) -> AgentResponse:
    # Returns: AgentResponse(answer=str, sources=list[str], confidence=float)
```

#### Step 4 — Hello World Retrieval Test (15 min)
Write a quick test script (not production code) that:
1. Creates a VectorStoreManager instance
2. Ingests the sample chunk from `examples/sample_chunk.json`
3. Queries for "what is a neural network"
4. Prints the retrieved chunk and its score

This proves end-to-end connectivity before the Prompt Engineer and UX Lead
need to integrate.

**Red flag:** If you are still in environment setup at the 40-minute mark,
ask the instructor immediately.

---

### UX Lead — Phase 1 Milestone
**Goal: Static UI layout running locally with all three panels visible**

#### Step 1 — Choose and Confirm Your Framework (5 min)
**Streamlit** — recommended for speed. Best if your team plans to deploy to
Streamlit Community Cloud.

**Gradio** — recommended if your team plans to deploy to HuggingFace Spaces.
Gradio's `gr.ChatInterface` gives you a production-quality chat component
with minimal code.

Both work. Pick one and do not switch mid-sprint.

#### Step 2 — Get the API Contract from the Pipeline Engineer (5 min)
Do not build any UI that calls the backend until you have the written API
contract from Step 3 of the Pipeline Engineer's Phase 1. Write it in a
comment at the top of `src/rag_agent/ui/app.py` so both roles stay aligned.

#### Step 3 — Build the Static Layout (20 min)
Build the three-panel layout with placeholder content. No backend calls yet.

**Panel 1 — Document Ingestion**
- Multi-file uploader accepting `.pdf` and `.md` files simultaneously
- Upload button that triggers ingestion (wired to backend in Phase 2)
- Status display: show ingestion results (files ingested, chunks created,
  duplicates skipped)
- List of currently ingested documents with document IDs

**Panel 2 — Document Viewer**
- Dropdown or list to select an ingested document
- Text/content display area showing the document content
- For PDFs: display extracted text. For markdown: render formatted.

**Panel 3 — Chat Interface**
- Chat history display (scrollable)
- Text input for user query
- Submit button
- Response display with **source citations visible** — every response must
  show which chunks it drew from
- Optional: topic filter dropdown (ANN, CNN, LSTM, etc.) to narrow retrieval
- Optional: difficulty filter (beginner, intermediate, advanced)

#### Step 4 — Local Run Check (10 min)
Confirm the static UI runs without errors:
```bash
# Streamlit
uv run streamlit run src/rag_agent/ui/app.py

# Gradio
uv run python src/rag_agent/ui/app.py
```

**Red flag:** If you are designing in Figma instead of writing UI code,
you are behind. Wireframes were for Hour 1.

---

### Prompt Engineer — Phase 1 Milestone (4+ person teams)
**Goal: All three core prompts drafted and manually validated**

#### Step 1 — Draft and Test Prompts Outside the Codebase (20 min)
Test all prompts directly in Claude, ChatGPT, or your chosen LLM before
touching code. Use the sample chunk from `examples/sample_chunk.json` as
your test context.

**Prompt 1 — System Prompt**
Define the agent's identity, constraints, and behavior. Must include:
- Role definition: interview preparation assistant for deep learning topics
- Instruction to answer only from retrieved context, not general knowledge
- Instruction to cite the source chunk for every factual claim
- Instruction to indicate when no relevant context was found rather than
  hallucinating
- Tone: clear, technical, like a senior engineer conducting a fair interview

**Prompt 2 — Question Generation Prompt**
Given a retrieved chunk, generate one interview question at a specified
difficulty level. Must produce:
- The question itself
- The difficulty level (beginner / intermediate / advanced)
- The topic tag
- A model answer drawn from the chunk
- One follow-up question to probe deeper

**Prompt 3 — Answer Evaluation Prompt**
Given a question, a student's answer, and the source chunk, evaluate the
answer. Must produce:
- Score out of 10
- What the student got right
- What was missing or incorrect
- The ideal answer based on the chunk

#### Step 2 — Document Expected I/O (10 min)
For each prompt, write the input format and expected output format in
`src/rag_agent/agent/prompts.py` as docstrings. The Pipeline Engineer
will implement the actual prompt templates from your documentation.

#### Step 3 — Identify Failure Modes (10 min)
For each prompt, write down one way it could fail and how you would fix it.
Example: "Question generation produces trivial yes/no questions — fix by
adding instruction to require open-ended questions with a minimum of two
concepts."

---

### QA and Interview Lead — Phase 1 Milestone (5-person teams)
**Goal: Test plan written, Hour 3 questions drafted, team risk assessment complete**

#### Step 1 — Write Your Test Plan (15 min)
Prepare these five test cases to run at the 60-minute mark of Hour 2:

| Test | Input | Expected Behavior |
|---|---|---|
| Normal query | "Explain the vanishing gradient problem" | Relevant chunks retrieved, accurate answer, source cited |
| Edge case query | "What is the meaning of life" | System responds that no relevant context was found — does NOT hallucinate |
| Duplicate ingestion | Upload the same file twice | Second upload is detected and skipped, user is notified |
| Empty query | Submit with blank input | Graceful error message, no crash |
| Cross-topic query | "How do LSTMs improve on RNNs for Seq2Seq tasks" | Chunks from multiple topics retrieved and synthesized |

#### Step 2 — Draft Hour 3 Interview Questions (15 min)
Prepare three technical interview questions your team will ask opponents.
Rules:
- Questions must be answerable from a well-built corpus
- Must require genuine understanding, not just recall
- At least one question must require connecting two topics
  (e.g., "How does the encoder in a Seq2Seq model relate to an autoencoder?")
- Prepare a model answer for each so you can judge responses fairly

#### Step 3 — Risk Assessment (10 min)
Review the competition rubric. For each category, write down your team's
current risk level (low / medium / high) and one action to reduce it.
Share this with the team before standup.

---

## Team Standup (0:40 – 0:45)

**5 minutes. Hard stop. Every team member speaks.**

Each person answers three questions only:
1. What do I have right now?
2. What do I need from another team member?
3. What is blocking me?

Instructor circulates during standup for F2F. Online teams post standup
notes in their team channel before moving to Phase 2.

**Common blockers to flag:**
- Corpus Architect: metadata schema not agreed yet → resolve with Pipeline Engineer now
- Pipeline Engineer: ChromaDB not initializing → check `chroma_db_path` in config
- UX Lead: not sure what the backend returns → get written API contract now
- Prompt Engineer: prompts not tested → spend first 10 min of Phase 2 on this only

---

## Phase 2 — Integration and Hardening (0:45 – 1:25)

Phase 2 is where the system becomes whole. Roles converge. The individual
components built in Phase 1 must now talk to each other.

---

### All Roles — Integration Priority Order

Work in this order to unblock the critical path:

**1. Pipeline Engineer + Corpus Architect (first 10 min of Phase 2)**
Run the first real ingestion. Feed the Corpus Architect's Phase 1 chunks
into the VectorStoreManager. Verify:
- Chunks are stored with correct metadata
- Duplicate detection fires correctly on a re-run
- Query returns ranked results with scores

**2. Pipeline Engineer + Prompt Engineer (next 10 min)**
Wire the prompts into the LangGraph nodes. The `generation_node` takes
retrieved chunks and the user query, applies the system prompt and question
generation prompt, and returns a structured response.

**3. Pipeline Engineer + UX Lead (ongoing)**
Replace all placeholder UI calls with real backend calls. The UX Lead
calls `ingest()`, `query()`, and `chat()` from the wired-up backend.
Source citations must appear in the chat response.

**4. QA Lead (final 20 min of Phase 2)**
Run all five test cases from the test plan. Log results. Flag any failures
to the relevant role owner immediately. Prioritise fixing the hallucination
guard and the duplicate detection — these are the most commonly tested
in interviews.

---

### Pipeline Engineer — Phase 2 Tasks

#### Implement the LangGraph Agent Graph
Open `src/rag_agent/agent/graph.py` and implement the state graph.

**Minimum viable graph — 3 nodes:**

```
[START] → query_rewrite_node → retrieval_node → generation_node → [END]
                                      ↑                  |
                                      └── (loop if no    |
                                           results found) |
```

**Node responsibilities:**

`query_rewrite_node` — Takes the raw user query and rewrites it to be
more retrieval-friendly. Example: "tell me about forgetting in LSTMs"
becomes "LSTM forget gate mechanism cell state." This node is a
great interview talking point.

`retrieval_node` — Calls `VectorStoreManager.query()`. If zero chunks
are returned above the similarity threshold, set a flag in state to
trigger the hallucination guard. Do not loop more than once.

`generation_node` — Takes retrieved chunks and constructs the final
response using the system prompt. If the hallucination guard flag is
set, return a "no relevant context found" message. Always include
source citations in the response.

#### Implement Conversation Memory
Use LangGraph's built-in checkpointer for conversation memory.
The `AgentState` in `src/rag_agent/agent/state.py` must include
`messages: list[BaseMessage]` so history is maintained across turns.
Implement token-aware trimming — when history approaches 3000 tokens,
trim the oldest non-system messages.

#### Implement the Full Query Pipeline
```python
# This is the flow the UI calls:
# 1. User submits query + history
# 2. Graph invokes: rewrite → retrieve → generate
# 3. Return answer + source citations + confidence indicator
```

---

### Corpus Architect — Phase 2 Tasks

#### Complete All Core Topics
Finish all six core topics with a minimum of 3 chunks each. Topics:
ANN, CNN, RNN, LSTM, Seq2Seq, Autoencoder.

If you finish early, add bonus topics (SOM, Boltzmann, GAN).

#### Ingest Landmark Paper PDFs
Work with the Pipeline Engineer to ingest at least two landmark paper
PDFs. Note that PDF chunking behaves differently from markdown — headers,
equations, and reference sections often produce noisy chunks. Review the
ingested PDF chunks and delete or edit any that are too noisy to be useful.

#### Corpus Quality Self-Check
Before handing off to QA, run this checklist:
- [ ] Every chunk passes the "remove the topic name" test
- [ ] Every chunk has complete, correct metadata
- [ ] No chunk is under 100 words or over 300 words
- [ ] Bonus topics flagged with `"is_bonus": true`
- [ ] At least two landmark paper PDFs ingested
- [ ] Related topics field populated for all chunks

---

### UX Lead — Phase 2 Tasks

#### Wire Up Ingestion
Replace the placeholder upload handler with a real call to the backend
ingestion pipeline. The UI must:
- Accept multiple files in a single upload event
- Show a progress indicator during ingestion
- Display ingestion results: files processed, chunks created, duplicates
  skipped, any errors
- Update the ingested documents list after successful ingestion

#### Wire Up the Document Viewer
After ingestion, the viewer panel must show:
- A selectable list of all ingested documents
- The content of the selected document rendered in the panel
- For landmark paper PDFs: show extracted text with page references

#### Wire Up the Chat Interface
Replace the placeholder chat with real calls to the LangGraph agent.
Every response displayed in the chat must include:
- The answer text
- Source citations: topic name + source file for each chunk used
- An indicator if no relevant context was found (do not silently return
  a hallucinated answer)

#### Streaming (stretch goal)
If time allows, implement streaming responses so the LLM output appears
token by token rather than waiting for full completion. This is a strong
interview talking point and a visible "wow factor" in Hour 3.

---

### Prompt Engineer — Phase 2 Tasks

#### Integrate Prompts into the Agent
Work with the Pipeline Engineer to move your tested prompts from
`prompts.py` into the LangGraph nodes. Verify each prompt:
- Produces well-structured output when given real retrieved chunks
- Handles edge cases: empty context, very short chunks, off-topic queries
- Generates questions at the requested difficulty level

#### Tune for Interview Quality
Run at least 10 manual test queries through the live system and evaluate:
- Are generated questions genuinely challenging?
- Do model answers accurately reflect the corpus?
- Are difficulty levels being correctly applied?
- Are source citations appearing correctly?

Iterate on prompts based on what you observe. Document your final prompt
versions clearly — you will be asked to explain your design choices in
Hour 3.

---

### QA Lead — Phase 2 Tasks (final 20 min)

#### Run the Full Test Plan
Execute all five test cases from your Phase 1 test plan against the
integrated system. For each test, record:
- Pass / Fail
- Actual behaviour observed
- Severity if failed (blocks demo / degrades demo / cosmetic)

**Critical failures that must be fixed before Hour 3:**
- System crashes on any test input
- Hallucination guard not working (system answers confidently with no
  relevant context)
- Duplicate ingestion not detected
- Source citations not appearing in responses

**Non-critical (flag but do not delay demo for):**
- Slow response time
- Cosmetic UI issues
- Bonus topics not yet ingested

#### Prepare the Demo Script
Write a 60-second demo script for Hour 3. The script must hit these
beats in order:
1. Show the document ingestion (upload two files)
2. Show duplicate detection (upload one of them again)
3. Show a normal query with source citation visible
4. Show the hallucination guard (query something off-topic)
5. Show a generated interview question with model answer

Practice this once before Hour 3 begins.

---

## Demo Rehearsal (1:25 – 1:35)

**10 minutes. Non-negotiable. Do not skip this.**

One full end-to-end run-through before Hour 3. The person who runs the
demo in Hour 3 must run it here first. The rest of the team watches and
notes anything that looks wrong, slow, or confusing.

This is not polish time. If something is broken, you have 10 minutes
to decide whether to fix it or work around it gracefully in the demo.
A clean workaround explained openly scores better in Hour 3 than a
hidden bug that surfaces during judging.

**Online teams:** Record a 2-minute Loom walkthrough of the working
system at the end of class. This is your async deliverable and your
fallback if anything breaks during the live presentation.

---

## Online Team — Async Continuation Checklist

For the week following class, use this as your completion checklist:

### By End of Day 1 (day of class)
- [ ] Starter repo cloned, environment running for all team members
- [ ] Roles confirmed and posted in team channel
- [ ] Architecture diagram committed to repo as `docs/architecture.md`
- [ ] API contract written and shared

### By Midweek
- [ ] All core topics drafted (ANN, CNN, RNN, LSTM, Seq2Seq, Autoencoder)
- [ ] At least two landmark papers located and downloaded
- [ ] VectorStoreManager ingestion and query working
- [ ] Static UI layout running locally

### By End of Week
- [ ] Full integration complete — ingestion, retrieval, chat all working
- [ ] All five QA test cases passing
- [ ] Bonus topics added if time allows
- [ ] Demo rehearsal recorded as Loom video
- [ ] Final version pushed to GitHub with clean commit history
- [ ] Deployed to Streamlit Community Cloud or HuggingFace Spaces

---

## Common Pitfalls and How to Handle Them

**"ChromaDB isn't persisting between runs"**
Check that `chroma_db_path` points to a real directory and that you are
using `PersistentClient`, not `EphemeralClient`. The path must exist before
initialization.

**"Embeddings are slow"**
If using a local embedding model via sentence-transformers, the first run
loads the model into memory. Subsequent runs are fast. Use
`@st.cache_resource` in Streamlit or equivalent caching to load the model
once per session.

**"LangGraph graph is not advancing past the first node"**
Check that your node functions return the updated state dict, not None.
Every node must explicitly return the state even if it only modifies one
field.

**"Duplicate detection is not working"**
Ensure your `doc_id` is generated deterministically from the file content
(e.g., a hash of the file contents), not from the filename or a timestamp.
Two uploads of the same file must produce the same `doc_id`.

**"The hallucination guard is triggering on everything"**
Your similarity threshold is too high. Start with 0.3 and tune upward.
Print the actual similarity scores during testing to calibrate.

**"Streamlit reruns the entire script on every interaction"**
Use `st.session_state` to store the VectorStoreManager instance, chat
history, and ingested document list. Without this, the ChromaDB connection
is re-initialized on every button click.
