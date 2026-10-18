from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
classifier_prompt = ChatPromptTemplate.from_template("""
You are a routing classifier.

Your task is ONLY to classify the user's message.

Possible labels:

GREETING
CHAT
DOCUMENT

GREETING:
- hello
- hi
- hey
- good morning

CHAT:
- thanks
- thank you
- goodbye
- okay
- nice
- how are you

DOCUMENT:
Everything else.

Examples:

Hi
GREETING

Hello
GREETING

Thanks
CHAT

How are you?
CHAT

Explain transformers.
DOCUMENT

Summarize chapter 3.
DOCUMENT

What models are used?
DOCUMENT

User:
{question}

Output ONLY one word:

GREETING
CHAT
DOCUMENT
""")

def classify_question(llm, question):

    chain = (
        classifier_prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({"question": question})

    result = result.strip().upper()

    if result not in {"GREETING", "CHAT", "DOCUMENT"}:
        result = "DOCUMENT"

    return result