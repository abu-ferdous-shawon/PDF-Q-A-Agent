from collections import Counter


def calculate_confidence(
    child_docs,
    parent_docs,
    compressed_docs,
    enough_context,
):
    """
    Returns a confidence score between 0 and 100.
    """

    score = 0

    # -----------------------------
    # Judge confidence
    # -----------------------------

    if enough_context:
        score += 40

    # -----------------------------
    # Retrieval depth
    # -----------------------------

    score += min(len(child_docs), 20)

    # -----------------------------
    # Parent diversity
    # -----------------------------

    score += min(len(parent_docs) * 5, 20)

    # -----------------------------
    # Compressed evidence
    # -----------------------------

    score += min(len(compressed_docs) * 5, 20)

    return min(score, 100)