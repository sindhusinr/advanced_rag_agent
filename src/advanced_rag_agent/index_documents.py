from advanced_rag_agent.ingestion.ingest import ingest_document


def main():

    ingest_document("data/sample.pdf")

    print("Indexing complete ✅")


if __name__ == "__main__":
    main()