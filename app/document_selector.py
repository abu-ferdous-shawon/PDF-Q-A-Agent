from difflib import SequenceMatcher


class DocumentSelector:

    def __init__(self, document_registry):
        self.registry = document_registry

    def select(self, question: str):

        question = question.lower().strip()

        selected = []

        for document_id, doc in self.registry.items():

            candidates = [
                doc.title or "",
                doc.source or "",
            ]

            matched = False

            for candidate in candidates:

                candidate = candidate.lower()

                # Exact substring
                if candidate and candidate in question:
                    matched = True
                    break

                # Question inside title
                if question in candidate:
                    matched = True
                    break

                # Token overlap
                question_words = {
                    w
                    for w in question.split()
                    if len(w) > 2
                }

                candidate_words = {
                    w
                    for w in candidate.split()
                    if len(w) > 2
                }

                overlap = len(
                    question_words &
                    candidate_words
                )

                if overlap >= 2:
                    matched = True
                    break

                # Fuzzy similarity
                score = SequenceMatcher(
                    None,
                    question,
                    candidate,
                ).ratio()

                if score >= 0.80:
                    matched = True
                    break

            if matched:
                selected.append(document_id)

        return selected