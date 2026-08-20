# retrieval/reranker.py

from advanced_rag_agent.generation.llm import (
    get_llm,
)


def rerank_documents(
    query,
    documents,
    top_k=5,
):

    llm = get_llm()

    scored_docs = []

    for doc in documents:

        prompt = f"""
Score the relevance of this chunk
to the question.

Question:
{query}

Chunk:
{doc.page_content}

Return only a number from 0 to 10.
"""

        try:

            score = llm.invoke(
                prompt
            ).content.strip()

            score = float(score)

        except Exception:

            score = 0

        scored_docs.append(
            (
                doc,
                score,
            )
        )

    ranked = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        doc
        for doc, _
        in ranked[:top_k]
    ]