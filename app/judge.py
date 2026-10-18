
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


judge_prompt = ChatPromptTemplate.from_template("""
You are evaluating whether retrieved PDF text contains
sufficient evidence to answer a user's question.

Question:
{question}

Retrieved Context:
{context}

Evaluation rules:

1. Determine whether the context directly addresses the
   user's actual question.

2. The answer does not need to appear as one exact sentence.
   It may be constructed from multiple supported passages.

3. For questions about document sections such as abstracts,
   conclusions, introductions, or methodologies, the context
   must contain the relevant section content, not merely
   a mention in the table of contents.

4. Do not confuse related information with the requested
   information.

5. Do not require exact wording when equivalent information
   clearly answers the question.

6. Never invent information that is absent from the context.

7. If the context contains enough evidence for a meaningful,
   supported answer, respond YES.

8. If important information is missing, respond NO.

Reply using exactly one word:

YES or NO
""")


def has_enough_context(llm, question, context):

    if not context or not context.strip():
        print("[Judge] Empty context.")
        return False

    chain = (
        judge_prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({
        "question": question,
        "context": context,
    })

    decision = result.strip().upper()

    decision = re.sub(
        r"[^A-Z]",
        "",
        decision
    )

    enough = decision == "YES"

    print(f"[Judge] Decision: {decision}")

    return enough
