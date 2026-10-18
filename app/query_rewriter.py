from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


rewrite_prompt = ChatPromptTemplate.from_template(
"""
You are an expert query rewriting assistant.

Your job is to rewrite the user's latest question into a
standalone question that contains all missing context.

Conversation History:
{history}

Current Question:
{question}

Rules:

- If the question is already standalone,
  return it unchanged.

- If it refers to previous messages
  using words like:

    it
    they
    them
    he
    she
    there
    that
    this
    those
    second model

Rewrite only when necessary.

Never paraphrase technical terms.

Never replace important nouns with synonyms.

Never change names.

Never change terminology.

If already standalone,
return it exactly unchanged.

Only return the rewritten question.

Do NOT answer it.
"""
)


def rewrite_question(llm, history, question):

    chain = (
        rewrite_prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "history": history,
            "question": question,
        }
    )