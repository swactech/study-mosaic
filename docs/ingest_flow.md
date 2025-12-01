# Ingestion Flow (Study-Mosaic)

```
User PDFs
   |
   v
Streamlit upload widget
   |
   v
Saved to: data/uploads/<session_id>/
   |
   v
pdfplumber extraction (page by page)
   |
   v
Chunking (chunk_size=800, overlap=120)
   |
   v
Chunk metadata:
  - chunk_id: <pdf-stem>-p<page>-c<idx>
  - page: page number
  - pdf: filename
  - char_start / char_end
   |
   v
Embeddings via Gemini (GEMINI_API_KEY)
   |
   v
Chroma local collection (name = session_id) @ data/vectors/
   |
   v
Ready for retrieval & grounded flashcards
```
