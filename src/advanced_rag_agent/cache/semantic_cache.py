from sklearn.metrics.pairwise import cosine_similarity

from advanced_rag_agent.ingestion.embedder import get_embedding_model


cache_store = []

embedding_model = get_embedding_model()


def get_cached_result(query: str, threshold: float = 0.90):

    query_embedding = embedding_model.embed_query(query)

    best_result = None
    best_score = 0

    for item in cache_store:

        score = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        if score > best_score:
            best_score = score
            best_result = item

    if best_result and best_score >= threshold:

        print(f"\n===== CACHE HIT ({best_score:.2f}) =====")

        return best_result["result"]

    print("\n===== CACHE MISS =====")

    return None


def save_to_cache(query: str, result):

    embedding = embedding_model.embed_query(query)

    cache_store.append(
        {
            "query": query,
            "embedding": embedding,
            "result": result
        }
    )