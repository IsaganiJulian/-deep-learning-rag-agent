# Deep Learning RAG Interview Prep Agent
### Graduate Data Science Capstone

A RAG-powered interview preparation agent built with LangChain, LangGraph,
and ChromaDB. Ingest deep learning study material and chat with it to
generate and answer technical interview questions.

---

## Quick Start

### Prerequisites
- Python 3.11 or 3.12
- [UV](https://docs.astral.sh/uv/) package manager
- VSCode (recommended)
- One of: Groq account (free), Ollama, or LM Studio

### Setup

```bash
# 1. Clone your team repo
git clone <your-team-repo-url>
cd deep-learning-rag-agent

# 2. Install dependencies with UV
uv sync

# 3. Copy and configure environment
cp .env.example .env
# Edit .env — choose your LLM provider and fill in credentials

# 4. Create required directories
mkdir -p data/corpus data/chroma_db

# 5. Verify setup
uv run python -c "import chromadb; import langchain; import langgraph; print('All dependencies OK')"

# 6. Run the app
uv run streamlit run src/rag_agent/ui/app.py
```

---

## LLM Provider Setup

### Option 1 — Groq (Recommended: free, fast, no local GPU needed)

1. Create a free account at [console.groq.com](https://console.groq.com)
2. Generate an API key under API Keys
3. In `.env`:
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```
4. Available models (free tier): `llama-3.1-8b-instant`, `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`

**Interview talking point:** Groq uses a custom LPU (Language Processing Unit)
chip designed specifically for LLM inference, delivering significantly lower
latency than GPU-based inference APIs.

---

### Option 2 — Ollama (Fully local, no API key, runs offline)

1. Download and install from [ollama.com](https://ollama.com)
2. Pull a model:
   ```bash
   ollama pull llama3.2        # 2B — fast, low memory
   ollama pull mistral         # 7B — better quality
   ollama pull llama3.1:8b     # 8B — good balance
   ```
3. Start the Ollama server (must be running before the app):
   ```bash
   ollama serve
   ```
4. In `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```

**Troubleshooting:** If the app cannot connect, verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

**Interview talking point:** Local inference eliminates data privacy concerns
and removes API cost entirely — important for enterprise deployments with
proprietary training data.

---

### Option 3 — LM Studio (Local GUI, OpenAI-compatible API)

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Open LM Studio, go to **Discover** and download a model
   (recommended: Llama 3.2 3B Instruct or Mistral 7B Instruct)
3. Go to **Local Server** tab → Load the model → Start Server
4. The server starts on `http://localhost:1234`
5. In `.env`:
   ```
   LLM_PROVIDER=lmstudio
   LMSTUDIO_BASE_URL=http://localhost:1234/v1
   LMSTUDIO_MODEL=local-model
   ```

**Interview talking point:** LM Studio exposes an OpenAI-compatible API,
which means any tooling written for OpenAI works without code changes —
just a `base_url` swap. This is the adapter pattern in practice.

---

## Deployment

### Streamlit Community Cloud (for Streamlit UI teams)

1. Push your code to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repo, branch, and `src/rag_agent/ui/app.py` as the main file
5. Under **Advanced Settings → Secrets**, add your environment variables
   (same keys as `.env`, one per line)
6. Click **Deploy**

Note: Streamlit Community Cloud does not persist local files between restarts.
For deployment, consider switching to a hosted ChromaDB instance or
committing a pre-built database to the repo.

### HuggingFace Spaces (for Gradio UI teams, or Streamlit teams)

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to Spaces → Create New Space
3. Choose **Streamlit** or **Gradio** as the SDK
4. Connect your GitHub repo or upload files directly
5. Add your API keys under **Settings → Repository Secrets**

HuggingFace Spaces provides 16GB RAM on the free CPU tier — better than
Streamlit Community Cloud for memory-intensive embedding operations.

---

## Project Structure

```
deep-learning-rag-agent/
├── pyproject.toml              # UV/pip dependencies and project config
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
│
├── data/
│   ├── corpus/                 # Corpus Architect: add .md and .pdf files here
│   └── chroma_db/              # Auto-created: local ChromaDB persistence
│
├── examples/
│   └── sample_chunk.json       # Canonical chunk schema reference
│
├── src/
│   └── rag_agent/
│       ├── config.py           # Settings, LLMFactory, EmbeddingFactory
│       │
│       ├── corpus/
│       │   └── chunker.py      # DocumentChunker — file loading and splitting
│       │
│       ├── vectorstore/
│       │   └── store.py        # VectorStoreManager — ChromaDB interface
│       │
│       ├── agent/
│       │   ├── state.py        # AgentState, data models
│       │   ├── prompts.py      # All LLM prompt templates
│       │   ├── nodes.py        # LangGraph node functions
│       │   └── graph.py        # Graph assembly and compilation
│       │
│       └── ui/
│           └── app.py          # Streamlit application
│
└── tests/
    └── test_vectorstore.py     # Unit tests for VectorStoreManager
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## Team Roles

| Role | Primary Files | Phase 1 Goal |
|---|---|---|
| Corpus Architect | `data/corpus/` | 3 topics drafted, schema agreed |
| Pipeline Engineer | `config.py`, `store.py`, `nodes.py`, `graph.py` | ChromaDB initialised, hello world retrieval |
| UX Lead | `ui/app.py` | Static three-panel layout running locally |
| Prompt Engineer | `prompts.py` | All three prompts drafted and manually tested |
| QA Lead | `tests/`, demo script | Test plan written, Hour 3 questions drafted |

---

## Corpus Requirements

### Core Topics (required)
- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Sequence-to-Sequence Models (Seq2Seq)
- Autoencoders

### Bonus Topics (differentiates exceptional teams)
- Self-Organizing Maps (SOM) — set `is_bonus: true` in metadata
- Boltzmann Machines — set `is_bonus: true`
- Generative Adversarial Networks (GAN) — set `is_bonus: true`

### Landmark Papers (minimum one per core topic)
See `docs/hour2_spec.md` for the full list with sources.

---

## Common Issues

**`ModuleNotFoundError: No module named 'rag_agent'`**
Run from the project root with `uv run`, not `python` directly.

**`chromadb.errors.NotEnoughElementsException`**
Your collection has fewer chunks than the requested `k`. Either ingest more
content or reduce `RETRIEVAL_K` in `.env`.

**`ollama: connection refused`**
Ollama is not running. Start it with `ollama serve` in a separate terminal.

**Streamlit reruns on every click and loses state**
Ensure all persistent objects (VectorStoreManager, graph) are wrapped in
`@st.cache_resource`. Conversation history must live in `st.session_state`.
