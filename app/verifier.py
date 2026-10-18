from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

verification_prompt = ChatPromptTemplate.from_template("""
You are checking whether an answer is fully supported by the retrieved context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Rules:

- Every factual claim in the answer must be traceable to the context.
- Paraphrasing and summarizing context content is allowed and is NOT a violation.
- Ignore writing style and grammar. Focus only on factual grounding.

Reply using ONLY

PASS

or

FAIL
""")


def verify_answer(llm, context, question, answer):

    chain = (
        verification_prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(
        {
            "context": context,
            "question": question,
            "answer": answer,
        }
    )

    return result.strip().upper().startswith("PASS")