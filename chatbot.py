from retriever import (create_retriever, retrieve_documents,)
from reranker import rerank_documents
from formatter import format_context
from prompts import MEDICAL_PROMPT, query_prompt
from llm import get_llm
from config import RERANK_THRESHOLD
from langchain_core.messages import AIMessage
from reranker import load_reranker


def rewrite_query(llm, query: str, history: str) -> str:

    if not history.strip():
        return query

    prompt = query_prompt.format(
        history=history,
        query=query
    )

    try:
        response = llm.invoke(prompt)

    except Exception:
        return query

    if isinstance(response.content, list):
        return response.content[0]["text"].strip()

    return response.content.strip()

def is_valid_query(query: str) -> bool:

    query = query.strip()

    # Empty query
    if not query:
        return False

    # No letters (only symbols/numbers)
    if not any(char.isalpha() for char in query):
        return False

    # Too much repeated text
    if len(set(query.replace(" ", ""))) <= 1:
        return False

    return True

def create_chatbot():

    retriever = create_retriever()

    llm = get_llm()

    load_reranker()

    def chatbot(query: str, history: str = ""):

        if not is_valid_query(query):
            return AIMessage(
                content="Please ask a clear medical question."
            )

        query = rewrite_query(llm, query, history)

        if query == "INVALID":
            return AIMessage(
                    content="I couldn't understand your question. Please ask a specific medical question."
            )

        documents = retrieve_documents(
            retriever,
            query
        )

        reranked = rerank_documents(query, documents)

        top_score = reranked[0][1]

        if top_score < RERANK_THRESHOLD:
            return AIMessage(
                "I couldn't find enough relevant medical information "
                "to answer your question. Please ask a more specific question."
        )

        documents = [
            doc
            for doc, score in reranked
        ]

        context = format_context(
            documents
        )

        prompt = MEDICAL_PROMPT.format(
            context=context,
            history=history,
            question=query
        )

        try:
            response = llm.invoke(prompt)

        except Exception:
            return AIMessage(
                content="The AI service is temporarily unavailable. Please try again later."
            )

        if isinstance(response.content, list):
            response.content = response.content[0]["text"]

        return response

    return chatbot


if __name__ == "__main__":

    chatbot = create_chatbot()

    while True:

        query = input("\nQuery (or 'exit'): ")

        if query.lower() == "exit":
            break

        response = chatbot(query)

        print("\n" + "=" * 80)
        print("Answer")
        print("=" * 80)

        print(response.content)