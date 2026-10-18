from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

hyde_prompt = ChatPromptTemplate.from_template("""
You are helping a retrieval system.

Expand the user's question into a search-oriented paragraph.

Rules:

- Keep all important keywords.
- Do NOT invent names.
- Do NOT invent facts.
- Do NOT answer the question.
- Simply rewrite it using related wording that might appear in the document.

Question:
{question}

Expanded Search Query:
""")


def generate_hypothetical_answer(llm, question):

    chain = (
        hyde_prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "question": question,
        }
    )