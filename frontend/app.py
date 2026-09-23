import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8080").rstrip("/")
print(f"Backend_url:{BACKEND_URL}")

st.set_page_config(page_title="CrewAI RAG Assistant", page_icon="🤖", layout="wide")

st.title("🤖 CrewAI RAG Assistant")
st.caption("Upload your document (PDF, TXT) and ask questions powered by CrewAI & FastAPI.")

# Sidebar: File Upload Section
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_file = st.file_uploader("Upload reference file", type=["pdf", "txt", "md"])

    if uploaded_file is not None:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Uploading and indexing into vector store..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    res = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f" Indexed {data.get('chunks_indexed', 0)} chunks!")
                    else:
                        st.error(f"Error: {res.json().get('detail', res.text)}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to FastAPI: {e}")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload a document on the left sidebar, and ask me any questions about it."}
    ]

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about your uploaded document..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("CrewAI agents are retrieving context and reasoning..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"query": prompt},
                    timeout=120
                )
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer received.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_detail = response.json().get("detail", response.text)
                    st.error(f"Error from server: {error_detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")