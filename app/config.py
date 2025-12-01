from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Centralized runtime configuration."""

    # Model configuration
    model_name: str = "gemini-2.5-flash"
    embedding_model: str = "models/embedding-001"

    # File system layout
    data_dir: Path = Path("data")
    vector_store_dir: Path = data_dir / "vectors"

    # Chunking defaults
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval defaults
    top_k: int = 6

    # Flashcard schema
    flashcard_schema_id: str = "https://example.com/flashcard.schema.json"

    # Security / keys
    api_key_env: str = "GEMINI_API_KEY"

    # Allowed PDFs per session
    max_pdfs_per_session: int = 5

    extra: dict = field(default_factory=dict)


settings = Settings()
