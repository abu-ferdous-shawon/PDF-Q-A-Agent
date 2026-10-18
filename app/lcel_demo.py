from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
    "Who is {person}?"
)

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke(
    {
        "person": "Alan Turing"
    }
)

print(response)