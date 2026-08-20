from sentence_transformers import (
    CrossEncoder,
)

from advanced_rag_agent.config.settings import (
    RERANK_MODEL,
)

reranker = CrossEncoder(
    RERANK_MODEL
)


def rerank_documents(
    query,
    documents,
    top_k=5,
):

    pairs = [
        (
            query,
            doc.page_content,
        )
        for doc in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(
            documents,
            scores,
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        doc
        for doc, _
        in ranked[:top_k]
    ]