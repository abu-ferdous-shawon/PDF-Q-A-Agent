from collections import deque


class ConversationMemory:
    def __init__(self, max_turns=5):
        self.history = deque(maxlen=max_turns)

    def add(self, question: str, answer: str):
        self.history.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    def clear(self):
        self.history.clear()

    def get_history(self):
        return list(self.history)

    def format(self):
        if not self.history:
            return ""

        lines = []

        for turn in self.history:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")

        return "\n".join(lines)