import json
from datetime import datetime


def log_retrieval(
    question,
    rewritten_question,
    hyde_query,
    child_docs,
    parent_docs,
    compressed_docs,
):
    log = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "rewritten_question": rewritten_question,
        "hyde_query": hyde_query,

        "child_documents": [
            {
                "page": d.metadata.get("page", -1) + 1,
                "source": d.metadata.get("source"),
                "parent_id": d.metadata.get("parent_id"),
            }
            for d in child_docs
        ],

        "parent_documents": [
            {
                "page": d.metadata.get("page", -1) + 1,
                "source": d.metadata.get("source"),
            }
            for d in parent_docs
        ],

        "compressed_documents": [
            {
                "page": d.metadata.get("page", -1) + 1,
                "source": d.metadata.get("source"),
            }
            for d in compressed_docs
        ],
    }

    with open("retrieval_logs.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")