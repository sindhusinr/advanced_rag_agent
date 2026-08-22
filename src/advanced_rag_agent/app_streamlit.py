import os
import uuid
import tempfile

import streamlit as st

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from advanced_rag_agent.graph.rag_graph import graph
from advanced_rag_agent.ingestion.ingest import ingest_document


# Cache graph so Streamlit doesn't rebuild it on every rerun
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


# Stores chat history displayed in UI
if "messages" not in st.session_state:
    st.session_state.messages = []


# Unique LangGraph conversation thread
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# Tracks active interrupt state
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = False


with st.sidebar:

    st.header("Agentic RAG")

    # Start a completely new conversation
    if st.button("🗑️ New Chat"):

        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending_interrupt = False

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

    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_pdf_path = temp_file.name

        with st.spinner("Indexing document..."):

            ingest_document(temp_pdf_path)

        st.success(
            f"{uploaded_file.name} indexed successfully!"
        )

        os.remove(temp_pdf_path)

    st.divider()

    st.markdown("### Session ID")

    st.code(
        st.session_state.thread_id
    )


# Render conversation history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==================================
# HANDLE RESUME AFTER INTERRUPT
# ==================================
if st.session_state.pending_interrupt:

    clarification = st.chat_input(
        "Provide additional information..."
    )

    if clarification:

        # Show user reply immediately
        with st.chat_message("user"):
            st.markdown(clarification)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": clarification
            }
        )

        with st.chat_message("assistant"):

            placeholder = st.empty()

            placeholder.markdown("Thinking...")

            result = graph_instance.invoke(
                Command(
                    resume=clarification
                ),
                config={
                    "configurable": {
                        "thread_id":
                        st.session_state.thread_id
                    }
                }
            )

            answer = "No response generated."

            if result.get("messages"):

                for msg in reversed(
                    result["messages"]
                ):

                    if (
                        hasattr(msg, "content")
                        and msg.content
                    ):
                        answer = msg.content
                        break

            placeholder.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.session_state.pending_interrupt = False

        st.rerun()


# ==================================
# NORMAL CHAT FLOW
# ==================================
else:

    question = st.chat_input(
        "Ask a question..."
    )

    if question:

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("assistant"):

            placeholder = st.empty()

            placeholder.markdown("Thinking...")

            result = graph_instance.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=question
                        )
                    ]
                },
                config={
                    "configurable": {
                        "thread_id":
                        st.session_state.thread_id
                    }
                }
            )

            if "__interrupt__" in result:

                interrupt_data = (
                    result["__interrupt__"][0]
                    .value
                )

                interrupt_question = (
                    interrupt_data["question"]
                )

                placeholder.markdown(
                    interrupt_question
                )

                # Save interrupt question into history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": interrupt_question
                    }
                )

                st.session_state.pending_interrupt = True

                st.rerun()

            answer = "No response generated."

            if result.get("messages"):

                for msg in reversed(
                    result["messages"]
                ):

                    if (
                        hasattr(msg, "content")
                        and msg.content
                    ):
                        answer = msg.content
                        break

            placeholder.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()