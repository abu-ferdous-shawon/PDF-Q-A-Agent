from sentence_transformers import CrossEncoder

# Load once when the application starts
model = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank(question, documents, top_k=5):
    """
    Rerank retrieved documents using a CrossEncoder.
    """

    if len(documents) == 0:
        return []

    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []

    for score, doc in ranked[:top_k]:

        doc.metadata["rerank_score"] = float(score)

        results.append(doc)

    return results