import pickle

from langchain_chroma import Chroma

from advanced_rag_agent.config.settings import (
    CHROMA_PATH
)


def create_vector_store(
    chunks,
    embeddings,
):

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    with open(
        "data/chunks.pkl",
        "wb",
    ) as file:

        pickle.dump(
            chunks,
            file,
        )

    return vector_store