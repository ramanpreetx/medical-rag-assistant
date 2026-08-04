from langchain_groq import ChatGroq
from config import MODEL_NAME
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():

    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=os.environ["GROQ_API_KEY"]
    )

    return llm

