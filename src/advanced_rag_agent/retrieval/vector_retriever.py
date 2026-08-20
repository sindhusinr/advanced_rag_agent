from langchain_chroma import Chroma

from advanced_rag_agent.config.settings import (
    CHROMA_PATH
)

from advanced_rag_agent.ingestion.embedder import (
    get_embedding_model
)

def get_retriever(k: int = 5):

    embeddings = get_embedding_model()

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )