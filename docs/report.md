### Problem Statement
Students and self-learners drown in dense PDFs (lectures, papers, manuals) and waste time turning them into study materials. Ungrounded AI summaries and flashcards hallucinate, eroding trust. The core problem: deliver high-quality, citation-backed study modules from PDFs quickly, without hallucinations, and make it repeatable across multiple PDFs and sessions.

### Why Agents?
Grounded study aids need orchestration: ingest once, retrieve relevant context, generate with strict policies, and validate coverage. Agents let us separate roles (generation vs coverage), enforce grounding prompts, and iterate until coverage goals are met. A supervisor agent coordinates retrieval + generation + coverage so the model stays on-task, transparent, and reliable—exactly what static scripts struggle to guarantee. Two techniques shine here:
- Multi-agent workflow: distinct flashcard and coverage agents, wrapped by a supervisor/loop agent to balance creativity with quality control.
- Context engineering: tight system prompts, schema-first responses, and chunk-level grounding/citations so the model never hallucinates outside provided context.

### What You Created
- Multi-agent scaffold: Supervisor agent orchestrates; Flashcard agent generates grounded cards; Coverage agent judges whether enough chunks are cited. The supervisor wraps them in a loop to stop only when coverage thresholds are met.
- RAG backbone: PDFs are parsed, chunked, embedded locally (Gemini embeddings), and stored in Chroma keyed by session. Retrieval feeds the generation context; citations are chunk- and page-referenced.
- Streamlit UI: Session-aware uploads with guardrails, ingest existing vs new uploads, sliders for coverage/max rounds, and JSON export with coverage stats. Bash-style flow hint at the top.
- Schema-first: Flashcard schema with verbatim citations and location metadata (page, chunk_id, char_start, char_end).

### Demo
Flow: `upload PDFs -> ingest (chunk+embed locally) -> generate flashcards (retrieval + coverage loop) -> download JSON`
- Upload 1–5 PDFs per session; ingest stores chunked text + metadata under `data/vectors/<session_id>/`.
- Generate flashcards: supervisor retrieves chunks, runs flashcard agent, coverage agent checks cited chunk_ids; loop continues until coverage threshold or max rounds.
- UI shows coverage summary and lets you download flashcards JSON with citations.

### The Build
- Stack: Google ADK (LlmAgent, LoopAgent), Gemini 2.5 (hosted), Chroma (local vector store), pdfplumber, Streamlit, dotenv.
- Ingestion: pdfplumber → chunker (size 800, overlap 120) → Gemini embeddings with retries → Chroma collection per session.
- Retrieval: semantic search over session collection with chunk metadata.
- Agents: Flashcard agent (grounded generation, schema), Coverage agent (continue/stop + missing chunks), Supervisor (retrieval + loop orchestration).
- UI: session-aware uploads, ingest guardrails (new vs existing), coverage controls, citation display, JSON export.

### If I Had More Time
- Add other generators (Socratic questions, summaries, quizzes) with shared coverage checks and switch to an ADK runner for richer logging and state.
- Rerank retrieval and cluster chunks to ensure topic coverage; add citation span validation in post-processing.
- Expand tests with PDF fixtures and integration checks for coverage/citations; add observability/logging on agent/tool calls.
- Offer session selection/history in the UI and expose download formats (CSV/Anki).***
