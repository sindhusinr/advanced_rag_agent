import os
import uuid
import tempfile

import streamlit as st

from langchain_core.messages import HumanMessage

from advanced_rag_agent.graph.rag_graph import graph
from advanced_rag_agent.ingestion.ingest import ingest_document


@st.cache_resource
def load_graph():
    return graph


graph_instance = load_graph()


st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


with st.sidebar:

    st.header("Agentic RAG")

    if st.button("🗑️ New Chat"):

        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()

    st.divider()

    st.markdown("### Model Settings")

    st.write("LLM: GPT OSS 120B")
    st.write("Embeddings: BGE Small")
    st.write("Reranker: MiniLM")

    st.divider()

    st.markdown("### Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:

            temp_file.write(uploaded_file.getvalue())

            temp_pdf_path = temp_file.name

        with st.spinner("Indexing document..."):

            ingest_document(temp_pdf_path)

        st.success(
            f"{uploaded_file.name} indexed successfully!"
        )

        os.remove(temp_pdf_path)

    st.divider()

    st.markdown("### Session")

    st.code(st.session_state.thread_id)


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


question = st.chat_input("Ask a question...")


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.spinner("Thinking..."):

        result = graph_instance.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }
        )

    answer = result["messages"][-1].content or "No response generated."

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)