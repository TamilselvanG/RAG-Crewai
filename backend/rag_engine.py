import os
import shutil
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool

# Persistent storage for Chroma
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
vector_store: Optional[Chroma] = None


def get_vector_store():
    global vector_store
    if vector_store is None:
        # Use open-source HuggingFace embedding model running locally
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},  # Change to "cuda" if using GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_store = Chroma(
            collection_name="rag_documents",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    return vector_store


def ingest_document(file_path: str) -> int:
    """Loads, splits, and embeds the uploaded document into ChromaDB."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    store = get_vector_store()
    # Add chunks to ChromaDB using HuggingFace embeddings
    store.add_documents(chunks)
    return len(chunks)


# Tool for CrewAI Agent to search the vector database
@tool("Search Vector Database")
def search_vector_db(query: str) -> str:
    """Searches the vector database for relevant document chunks matching the user's query."""
    store = get_vector_store()
    results = store.similarity_search(query, k=4)
    if not results:
        return "No relevant context found in the uploaded documents."
    return "\n\n---\n\n".join([doc.page_content for doc in results])


async def execute_rag_crew(user_query: str, 
                           enable_web_search: bool = False,) -> str:
    """Runs a CrewAI crew composed of a Retriever & Synthesizer to answer questions."""
    # Use native CrewAI LLM object
    llm = LLM(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    # 1. Retrieval Agent
    retrieval_agent = Agent(
        role="Information Retrieval Specialist",
        goal="Extract the most accurate and relevant facts from the knowledge base using vector search.",
        backstory="You are an expert at querying document stores and pulling precise context.",
        tools=[search_vector_db],
        llm=llm,
        verbose=False,
    )

    # 2. Answer Generator Agent
    synthesizer_agent = Agent(
        role="Knowledge Synthesizer & Writer",
        goal="Provide clear, concise, and well-grounded answers based strictly on retrieved context.",
        backstory="You synthesize information into clear, structured, natural answers. If information is missing, you state that transparently.",
        llm=llm,
        verbose=False,
    )

    # Tasks
    search_task = Task(
        description=f"Search the database for content related to: '{user_query}' and extract the key points.",
        expected_output="A summary of raw facts and excerpts answering the query.",
        agent=retrieval_agent,
    )

    web_search_status = (
        "Web search was requested, but it is not implemented yet."
        if enable_web_search
        else "Web search is disabled."
    )

    answer_task = Task(
        description=(f"Using the gathered facts, write a comprehensive and clear answer to: '{user_query}'. "
                     f"Do not hallucinate facts outside the provided document context. "
                     f"{web_search_status}"
        ),
        expected_output="A helpful, factual, markdown-formatted response.",
        agent=synthesizer_agent,
    )

    crew = Crew(
        agents=[retrieval_agent, synthesizer_agent],
        tasks=[search_task, answer_task],
        process=Process.sequential,
    )

    result = await crew.kickoff_async()
    
    return str(result)