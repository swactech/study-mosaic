### Problem Statement

Students and self-learners are surrounded by dense PDFs—lecture slides, research papers, manuals, standards documents—but turning them into usable study material is still a manual grind. Existing “AI flashcard” tools often feel like guessing machines: they summarize without clear references, mix concepts across pages, and hallucinate details, which quietly erodes trust and wastes even more time.

The problem this agent tackles is: **how to transform PDFs into high-quality, citation-backed study modules—quickly, repeatedly, and with verifiable grounding.** The agent needs to (1) ingest PDFs once, (2) target the right sections when generating flashcards, (3) attach precise citations, and (4) track how much of the document has been covered so learners can see where their study deck comes from and what remains.

---

### Why Agents?

This is more than “call an LLM with a prompt.” Reliable, grounded study assistance requires **orchestration** across several steps:

* ingest and index PDFs once per session,
* retrieve only the most relevant chunks for a given generation round,
* generate strictly grounded flashcards that must cite real text spans, and
* **evaluate coverage** to decide whether to continue or stop.

A single prompt or static script can’t easily enforce these policies or reason about coverage over time. Agents let us split responsibilities into clear roles (generation vs. coverage vs. coordination), enforce strict grounding prompts, and loop until coverage goals are met.

A **supervisor agent** coordinates retrieval, flashcard generation, and coverage checking. It decides when to call which agent, when to request more flashcards, and when to stop. This keeps the system on-task, transparent, and repeatable across multiple PDFs and study sessions—exactly the kind of multi-step, tool-using workflow the course emphasizes.

---

### What You Created

* **Multi-agent workflow with explicit supervision**

  * A **Supervisor agent** orchestrates the entire pipeline: it retrieves chunks, calls the Flashcard agent, queries the Coverage agent, and runs an iterative coverage loop. It owns the “policy” for when to continue or stop.
  * A **Flashcard agent** generates grounded flashcards from retrieved chunks. It must follow a schema, include citations, and avoid using knowledge outside the provided context.
  * A **Coverage agent** inspects the generated flashcards, checks which chunk IDs were cited, and estimates how much of the ingested context has been covered. It returns “continue/stop” plus hints about under-represented areas.
    Together, they turn the vague task “make flashcards from this PDF” into a structured, monitorable workflow.

* **Deliberate context engineering with RAG and grounding**

  * PDFs are parsed, chunked, and embedded locally using Gemini embeddings, then stored in a **Chroma** collection keyed by session ID.
  * Retrieval runs semantic search over the session’s chunks, feeding only relevant pieces into the Flashcard agent.
  * Each flashcard carries **chunk- and page-level references** so learners can trace every card back to its source.

* **Session-aware Streamlit UI**

  * A lightweight UI lets users create a “study session,” upload one or more PDFs, and either **reuse existing ingests** or process new files.
  * Sliders control coverage targets (e.g., “aim for 60–80% chunk coverage”) and maximum rounds, so users can trade off depth vs. speed.
  * After generation, the UI surfaces **coverage stats** (e.g., fraction of chunks cited, number of rounds used) and offers a JSON export of the flashcards.

* **Schema-first design**

  * Flashcards follow a strict JSON schema that includes:

    * question, answer, optional hints
    * **verbatim citation snippets**
    * location metadata: `page`, `chunk_id`, `char_start`, `char_end`.
  * This makes the output easy to inspect, visualize, or plug into other tools (Anki import, quizzes, etc.) while preserving grounding.

---

### Demo

A typical flow looks like this:

1. **Upload PDFs**

   * The user uploads 1–5 PDFs (e.g., a lecture pack plus a key reference paper). The app associates them with a new `session_id`.

2. **Ingest (chunk + embed locally)**

   * The ingestion pipeline parses each PDF, extracts text, chunks it, embeds chunks using Gemini embeddings, and stores them in `data/vectors/<session_id>/` as a Chroma collection with rich metadata.
   * On later visits, the same session can reuse these vectors without re-processing the PDFs.

3. **Generate flashcards via retrieval + coverage loop**

   * The Supervisor agent retrieves a batch of relevant chunks, calls the Flashcard agent to generate cards grounded in those chunks, then calls the Coverage agent to evaluate cited `chunk_id`s.
   * If coverage is below the user-defined threshold and the max round count isn’t reached, the supervisor requests another round focused on under-covered chunks.
   * This loop continues until **coverage goals or safety limits** are met.

4. **Review & export**

   * The UI shows a coverage summary: how many chunks were cited, approximate coverage percentage, and how many rounds were needed.
   * The user can download the **flashcards JSON with embedded citations** and integrate it into their study workflow.

---

### The Build

* **Stack**

  * **Agent framework:** Google ADK (e.g., `LlmAgent`, `LoopAgent`) to define the Supervisor, Flashcard, and Coverage agents and wire their calls.
  * **Models:** Gemini 2.5 (hosted) for both embeddings and generation.
  * **Vector store:** Chroma, running locally per session.
  * **PDF & utilities:** `pdfplumber` for robust text extraction, `Streamlit` for the UI, `dotenv` for configuration and API keys.

* **Ingestion pipeline**

  * `pdfplumber` extracts clean text and page boundaries.
  * A simple chunker (size 800, overlap 120) turns each document into overlapping semantic chunks while preserving page numbers.
  * Chunks are embedded with Gemini embeddings (with retries/backoff for robustness) and written into a **per-session Chroma collection**, storing metadata like `page`, `chunk_id`, and character offsets.

* **Retrieval**

  * Semantic search runs over the session’s collection, returning top-K chunks plus metadata to the Supervisor.
  * Retrievers can be tuned for “broad coverage” (cover more topics) vs. “deep focus” (drill into a section), which is where future reranking/cluster strategies plug in.

* **Agents**

  * **Flashcard agent:**

    * Input: retrieved chunks + instructions + schema.
    * Behavior: generate flashcards that **must** quote from the provided chunks and attach correct metadata.
  * **Coverage agent:**

    * Input: list of generated flashcards and the full set of chunk IDs.
    * Behavior: compute coverage (which chunk IDs were cited), identify gaps, and return a "continue/stop" decision plus hints about missing areas.
  * **Refiner agent:**

    * Input: prior flashcards plus coverage feedback and under-covered chunk hints.
    * Behavior: revise/augment flashcards to address gaps surfaced by the coverage agent before the next loop.
  * **Supervisor:**

    * Implements a small policy loop: retrieve -> generate -> evaluate coverage -> repeat or stop.
    * Orchestrates via an ADK `SequentialAgent` containing the flashcard agent and a `LoopAgent` for coverage/refinement; `_run_async_impl` delegates to that sequence agent (no abstract base calls) and the fallback import now includes `SequentialAgent`.
    * Tracks rounds, enforces safety limits, and logs simple metrics that could be expanded into full evaluation later.

* **UI**

  * Session-aware file uploads that distinguish between **new ingestion** and **reusing existing vectors**, to reduce latency and cost.
  * Guardrails that prevent generation before ingestion is complete and prevent accidental mixing of sessions.
  * Controls for coverage threshold, max rounds, and display of citation snippets so users can sanity-check grounding.
  * One-click JSON export for downstream tools.

---

### If I Had More Time

* **Richer study module types**

  * Add additional generators—Socratic question chains, concept summaries, spaced-repetition friendly decks, and practice quizzes—while reusing the same **coverage agent** so every module type remains grounded and coverage-aware.
  * Move orchestration into an ADK runner for richer logging, traces, and better visibility into each agent/tool call.

* **Smarter retrieval & coverage**

  * Introduce reranking and clustering over chunks to ensure **balanced topic coverage** instead of repeatedly sampling “easy” sections.
  * Add post-processing to validate citation spans (e.g., check that answer text actually appears in the cited span or nearby context).

* **Evaluation, testing, and observability**

  * Build a small test suite with PDF fixtures (lectures, papers, technical manuals) plus expected coverage ranges and sanity checks on citations.
  * Add observability around agent/tool calls (latency, failure modes, number of rounds per session) to guide future optimization.

* **User experience and persistence**

  * Support **session history** in the UI so learners can revisit past decks, compare coverage across versions, and download in multiple formats (JSON, CSV, Anki-ready).
  * Experiment with lightweight user profiles or per-course sessions so the agent can adapt coverage depth and question difficulty over time.
