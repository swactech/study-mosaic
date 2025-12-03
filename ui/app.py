"""
Minimal Streamlit UI for Study-Mosaic.
"""

import json
import sys
from pathlib import Path
from typing import List
import pandas as pd

import streamlit as st  # type: ignore
from dotenv import load_dotenv

# Ensure the project root is on sys.path when launched via `streamlit run ui/app.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.supervisor import SupervisorAgent  # noqa
from app.tools.ingest import ingest_pdfs  # noqa
from app.tools.vector_store import LocalVectorStore  # noqa

# Load environment variables from .env at repo root so Streamlit picks up keys.
load_dotenv(ROOT / ".env")


st.set_page_config(page_title="Study-Mosaic", layout="wide")
st.title("Study-Mosaic: one source, many learning shapes")
st.code(
    "upload PDFs -> ingest (chunk+embed locally) -> generate flashcards/socratic questions/summaries/quizzes (retrieval + coverage loop)",
    language="bash",
)


def save_uploads(uploaded_files, session_dir: Path) -> List[Path]:
    """Persist uploaded files to disk so the ingestion pipeline can read them."""
    saved_paths: List[Path] = []
    session_dir.mkdir(parents=True, exist_ok=True)
    for file in uploaded_files:
        path = session_dir / file.name
        with path.open("wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(path)
    return saved_paths


def get_saved_uploads(session_id: str) -> List[Path]:
    """Return existing uploaded PDFs for a session if present."""
    session_dir = Path("data/uploads") / session_id
    if not session_dir.exists():
        return []
    return sorted(session_dir.glob("*.pdf"))


def get_ingested_chunk_metadata(session_id: str) -> List[dict]:
    """Fetch stored chunk metadata for the session to inform generation UI."""
    try:
        store = LocalVectorStore(session_id=session_id)
        return store.list_metadatas()
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.warning(f"Unable to read ingested chunks for this session: {exc}")
        return []


def render_ingested_summary(chunk_metadata: List[dict]) -> None:
    """Show users what has been ingested so far."""
    if not chunk_metadata:
        st.info(
            "No ingested chunks yet for this session. Ingest PDFs to enable flashcard generation.")
        return

    st.success(
        f"{len(chunk_metadata)} ingested chunk(s) ready for flashcard generation.")

    by_pdf = {}
    for meta in chunk_metadata:
        pdf = meta.get("pdf", "Unknown PDF")
        entry = by_pdf.setdefault(pdf, {"chunks": 0, "pages": set()})
        entry["chunks"] += 1
        page = meta.get("page")
        if page is not None:
            entry["pages"].add(page)

    by_pdf_df = pd.DataFrame.from_dict(
        by_pdf, orient="index").reset_index().rename(columns={"index": "title"})
    # strip ".pdf" and optionally shorten for display
    by_pdf_df["title"] = by_pdf_df["title"].str.replace(
        ".pdf", "", regex=False)

    # convert the pages set to "min-max" string like "1-55"
    by_pdf_df["pages"] = by_pdf_df["pages"].apply(
        lambda s: f"{min(s)}-{max(s)}" if s else ""
    )
    st.dataframe(by_pdf_df)


def main():
    session_id = st.text_input("Session ID", value="demo-session")
    uploads = st.file_uploader(
        "Upload 1-5 PDFs", type=["pdf"], accept_multiple_files=True)
    saved = get_saved_uploads(session_id)
    if saved:
        st.caption("Found existing uploads for this session: " +
                   ", ".join(p.name for p in saved))

    col1, col2 = st.columns(2)
    with col1:
        ingest_new = st.button("Ingest NEW uploads")
    with col2:
        if saved:
            ingest_existing = st.button("Ingest EXISTING uploads")

    if ingest_new:
        if not uploads:
            st.error("Please upload 1-5 PDFs before ingesting new files.")
        else:
            saved_paths = save_uploads(
                uploads, Path("data/uploads") / session_id)
            with st.spinner("Ingesting PDFs..."):
                chunks = ingest_pdfs(session_id=session_id,
                                     pdf_paths=saved_paths)
            st.success(
                f"Ingested {len(chunks)} chunks across {len(saved_paths)} file(s).")

    if saved and ingest_existing:
        if not saved:
            st.error("No existing uploads found for this session.")
        elif uploads:
            st.warning(
                "You have new uploads selected. Ingesting existing files only. Use 'Ingest NEW uploads' for the new files."
            )
        else:
            with st.spinner("Ingesting existing PDFs..."):
                chunks = ingest_pdfs(session_id=session_id, pdf_paths=saved)
            st.success(
                f"Ingested {len(chunks)} chunks across {len(saved)} existing file(s).")

    prompt = st.text_input("Flashcard request",
                           value="Create flashcards for the whole document")
    coverage_threshold = st.slider("Coverage threshold", 0.1, 1.0, 0.8, 0.05)
    max_rounds = st.slider("Max rounds", 1, 5, 3, 1)

    ingested_metadata = get_ingested_chunk_metadata(session_id)
    render_ingested_summary(ingested_metadata)

    if ingested_metadata and st.button("Generate flashcards"):
        with st.spinner("Generating..."):
            supervisor = SupervisorAgent(
                max_iterations=max_rounds, coverage_threshold=coverage_threshold)
            result = supervisor.handle(
                session_id=session_id,
                message=prompt,
                top_k=None,
            )
        cards = result.get("flashcards", [])
        st.subheader(f"Flashcards ({len(cards)})")
        for card in cards:
            with st.expander(card.get("question", "Untitled")):
                st.markdown(f"**Q:** {card.get('question')}")
                st.markdown(f"**A:** {card.get('answer')}")
                st.markdown("**Citations:**")
                for cite in card.get("citations", []):
                    loc = cite.get("location", {})
                    st.code(
                        f"{cite.get('text')}\n(page {loc.get('page')} | chunk {loc.get('chunk_id')})"
                    )
        st.caption(
            f"Coverage: {result.get('coverage', 0):.2f} "
            f"({len(result.get('cited_chunks', []))}/{result.get('total_chunks', 0)} chunks cited)"
        )
        st.download_button(
            "Download JSON",
            data=json.dumps(cards, indent=2),
            file_name="flashcards.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
