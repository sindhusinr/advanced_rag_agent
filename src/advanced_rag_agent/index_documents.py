from advanced_rag_agent.ingestion.loader import (
    load_pdf,
)

from advanced_rag_agent.ingestion.chunker import (
    chunk_documents,
)

from advanced_rag_agent.ingestion.embedder import (
    get_embedding_model,
)

from advanced_rag_agent.ingestion.indexer import (
    create_vector_store,
)


def main():

    documents = load_pdf(
        "data/sample.pdf"
    )

    chunks = chunk_documents(
        documents
    )

    embeddings = get_embedding_model()

    create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("Indexing complete ✅")


if __name__ == "__main__":
    main()
