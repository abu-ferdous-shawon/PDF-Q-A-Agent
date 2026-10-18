from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
)

response = llm.invoke("Say hello in one sentence.")

print(response.content)