def chunk_text(text: str) -> list[str]:
    lines = text.splitlines()

    chunks = []
    current_chunk = []

    for line in lines:

        # Start a new chunk only for section headings
        if line.startswith("## "):

            # Save the previous chunk only if it contains
            # actual content, not just a heading
            if current_chunk:
                chunk = "\n".join(current_chunk).strip()

                # Don't save useless standalone headings
                content_lines = [
                    line for line in current_chunk
                    if line.strip() and not line.startswith("#")
                ]

                if content_lines:
                    chunks.append(chunk)

            # Start the new section
            current_chunk = [line]

        else:
            current_chunk.append(line)

    # Save the final chunk
    if current_chunk:
        content_lines = [
            line for line in current_chunk
            if line.strip() and not line.startswith("#")
        ]

        if content_lines:
            chunk = "\n".join(current_chunk).strip()
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    from src.loader import load_document

    text = load_document("data/hr_documents/leave_policy.md")
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk)