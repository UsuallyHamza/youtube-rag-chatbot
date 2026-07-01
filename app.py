# -*- coding: utf-8 -*-

import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import re

# Streamlit automatically maps secrets from the dashboard into st.secrets
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")

if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
else:
    st.warning("Missing HF_TOKEN. Downloads may be rate-limited.")

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="📺")

# -----------------------------------------
# 2. Core Logic Functions
# -----------------------------------------
def extract_video_id(url):
    """Extracts the 11-character YouTube video ID from standard or shortened URLs."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

@st.cache_resource(show_spinner=False)
def process_video(video_url):
    """Handles the downloading, splitting, and embedding.
    @st.cache_resource ensures this only runs ONCE per unique URL."""
    
    video_id = extract_video_id(video_url)
    if not video_id:
        return None, "Invalid YouTube URL. Please check the link."

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join([chunk["text"] for chunk in transcript_list])
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"
    
    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embedding and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, "Success"


def format_docs(retrieved_docs):
    """Combines retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def append_bge_prefix(question):
    """BGE models require a specific prefix for search queries to maximize accuracy."""
    return f"Represent this sentence for searching relevant passages: {question}"


"""# **Step 1a - Indexing (Document Ingestion)**"""



"""# **Step 1B - Indexing (Text Splitting)**"""


"""# **Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)**"""


"""# **Step2 - Retrival**"""


"""# **Step 3 - Augmentation**"""


"""# **Step 4 - Generation**"""


"""# **Building a Chain**"""


# -----------------------------------------
# 3. Streamlit UI & State Management
# -----------------------------------------
st.title("📺 YouTube RAG Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for inputs
with st.sidebar:
    st.header("Video Setup")
    url_input = st.text_input("Paste YouTube URL:")
    process_button = st.button("Process Video")

# Process the video when the button is clicked
if process_button and url_input:
    with st.spinner("Downloading and embedding transcript..."):
        vector_store, status_message = process_video(url_input)
        
        if vector_store:
            st.session_state.vector_store = vector_store
            st.success("Video processed successfully! You can now ask questions.")
        else:
            st.error(status_message)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Handler
if user_query := st.chat_input("Ask something about the video..."):
    # Ensure a video has been processed first
    if "vector_store" not in st.session_state:
        st.warning("Please process a YouTube video first.")
    else:
        # 1. Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Build the LangChain RAG Pipeline
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite', temperature=0.2)

        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            Context:
            {context}

            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        # This chain routes the modified query to the retriever, 
        # but passes the original query to the LLM prompt.
        parallel_chain = RunnableParallel({
            "context": RunnableLambda(append_bge_prefix) | retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        main_chain = parallel_chain | prompt | llm | StrOutputParser()

        # 3. Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = main_chain.invoke(user_query)
                st.markdown(response)
        
        # 4. Save to history
        st.session_state.messages.append({"role": "assistant", "content": response})