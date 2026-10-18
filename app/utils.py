def format_docs(docs):

    formatted = []

    for doc in docs:

        page = doc.metadata.get("page", -1) + 1

        source = doc.metadata.get("source", "Unknown")

        formatted.append(
            f"[{source} | Page {page}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)