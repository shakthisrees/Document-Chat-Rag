import os
import base64
import gc
import tempfile
import uuid

import streamlit as st

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


# Session initialization
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id


# Load LLM
@st.cache_resource
def load_llm():
    llm = Ollama(model="gemma3:4b", request_timeout=120.0)
    return llm


# Reset chat
def reset_chat():
    st.session_state.messages = []
    gc.collect()


# Display uploaded PDF
def display_pdf(file):
    st.markdown("### PDF Preview")

    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}"
    width="100%" height="600px" type="application/pdf">
    </iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)


# Sidebar
with st.sidebar:

    st.header("Add your documents")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:

        try:

            with tempfile.TemporaryDirectory() as temp_dir:

                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"

                st.write("Indexing document...")

                if file_key not in st.session_state.get("file_cache", {}):

                    loader = SimpleDirectoryReader(
                        input_dir=temp_dir,
                        required_exts=[".pdf"],
                        recursive=True
                    )

                    docs = loader.load_data()

                    llm = load_llm()

                    # Faster embedding model
                    embed_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-small-en-v1.5"
                    )

                    Settings.embed_model = embed_model

                    index = VectorStoreIndex.from_documents(
                        docs,
                        show_progress=True
                    )

                    Settings.llm = llm

                    query_engine = index.as_query_engine(streaming=True)

                    # Custom prompt
                    qa_prompt = PromptTemplate(
                        """
Context information is below.
---------------------
{context_str}
---------------------

Use the context above to answer the question.
If the answer is not found, say "I don't know".

Question: {query_str}

Answer:
"""
                    )

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt}
                    )

                    st.session_state.file_cache[file_key] = query_engine

                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Ready to chat!")

                display_pdf(uploaded_file)

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()


# Header
col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat with Docs using Gemma 3")

with col2:
    st.button("Clear", on_click=reset_chat)


# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Show chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask a question about your document"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if "query_engine" not in locals() and not st.session_state.file_cache:
        st.error("Please upload a PDF first.")
        st.stop()

    # Get query engine
    if "query_engine" not in locals():
        query_engine = list(st.session_state.file_cache.values())[0]

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""

        streaming_response = query_engine.query(prompt)

        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )