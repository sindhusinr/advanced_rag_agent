from langchain_huggingface import HuggingFaceEmbeddings

from advanced_rag_agent.config.settings import EMBEDDING_MODEL


embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


def get_embedding_model():

    return embedding_model