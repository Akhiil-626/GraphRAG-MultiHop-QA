from app.chunking import chunk_text


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    chunks = chunk_text(text, max_chars=120, overlap=1)
    assert len(chunks) > 1
    assert chunks[0].split(". ")[-1].strip() in chunks[1]
