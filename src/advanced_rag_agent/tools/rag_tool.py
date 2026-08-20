from langchain_core.tools import tool

from advanced_rag_agent.retrieval.hybrid_retriever import hybrid_search
from advanced_rag_agent.retrieval.reranker import rerank_documents
from advanced_rag_agent.cache.semantic_cache import get_cached_result, save_to_cache


@tool
def rag_tool(query: str) -> dict:
    """
    Search the uploaded documents and return relevant context.

    Use this tool when the user asks questions about:
    - Uploaded PDFs
    - Stored documents
    - ChunkedTejas
    - Concepts present in the knowledge base
    - Information that requires document retrieval

    Args:
        query: User question to search for.

    Returns:
        Dictionary containing retrieved content with citations.
    """

    cached_result = get_cached_result(query)

    if cached_result:
        return cached_result

    print("\n===== RAG TOOL CALLED =====")

    documents = hybrid_search(query=query, k=5)

    reranked_docs = rerank_documents(query=query, documents=documents, top_k=5)

    citations = []

    for doc in reranked_docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")

        citations.append(f"[Page {page}] {doc.page_content}")

    result = {"citations": citations}

    save_to_cache(query, result)

    return result