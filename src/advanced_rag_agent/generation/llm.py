from langchain_groq import ChatGroq

from advanced_rag_agent.config.settings import (
    GROQ_API_KEY,
    LLM_MODEL,
)


def get_llm():

    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,

        # Enable token streaming
        streaming=True,
    )