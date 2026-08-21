from advanced_rag_agent.ingestion.loader import load_pdf
from advanced_rag_agent.ingestion.chunker import chunk_documents
from advanced_rag_agent.ingestion.embedder import get_embedding_model
from advanced_rag_agent.ingestion.indexer import create_vector_store


def ingest_document(pdf_path: str):

    documents = load_pdf(pdf_path)

    chunks = chunk_documents(documents)

    embeddings = get_embedding_model()

    create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
    )