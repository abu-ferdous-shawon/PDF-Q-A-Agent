from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def compress(question, documents, top_k_paragraphs=5):

    compressed = []

    for doc in documents:

        text = doc.page_content.strip()

        # Tables must stay intact -- paragraph-level compression
        # would scramble row/column alignment.
        if doc.metadata.get("content_type") == "table":
            compressed.append(
                Document(
                    page_content=text,
                    metadata=doc.metadata.copy(),
                )
            )
            continue

        # ------------------------------------
        # Split intelligently
        # ------------------------------------

        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        # If the PDF contains no paragraph breaks,
        # keep the entire chunk.
        if len(paragraphs) <= 1:

            compressed.append(
                Document(
                    page_content=text,
                    metadata=doc.metadata.copy(),
                )
            )

            continue

        pairs = [
            (question, p)
            for p in paragraphs
        ]

        scores = model.predict(pairs)

        ranked = sorted(
            zip(scores, paragraphs),
            key=lambda x: x[0],
            reverse=True,
        )

        selected = [
            paragraph
            for _, paragraph in ranked[:top_k_paragraphs]
        ]

        compressed.append(
            Document(
                page_content="\n\n".join(selected),
                metadata=doc.metadata.copy(),
            )
        )

    return compressed