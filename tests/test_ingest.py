from app.tools.ingest import _chunk_text


def test_chunk_text_overlap():
    text = "abcdefghij" * 10  # 100 chars
    chunks = _chunk_text(text, chunk_size=20, overlap=5)
    # Ensure coverage and overlap behavior
    assert chunks, "Chunking should return at least one chunk"
    assert chunks[0]["char_start"] == 0
    assert chunks[0]["char_end"] == 20
    # Overlap means second chunk starts before first ends
    assert chunks[1]["char_start"] == 15
    assert chunks[1]["char_end"] == 35
    # Last chunk ends at text length
    assert chunks[-1]["char_end"] == len(text)
