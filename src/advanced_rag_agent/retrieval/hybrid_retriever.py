from advanced_rag_agent.retrieval.vector_retriever import get_retriever
from advanced_rag_agent.retrieval.bm25_retriever import create_bm25_retriever
from advanced_rag_agent.retrieval.chunk_store import load_chunks


def hybrid_search(query: str, k: int = 5):
    chunks = load_chunks()
    vector_retriever = get_retriever(k)
    bm25_retriever = create_bm25_retriever(chunks, k)
    vector_results = vector_retriever.invoke(query)
    bm25_results = bm25_retriever.invoke(query)

    combined = vector_results + bm25_results
    unique_docs = {}

    for doc in combined:
        unique_docs[doc.page_content] = doc

    return list(unique_docs.values())