# 📺 YouTube RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) web application that allows users to chat with the contents of any English YouTube video. 

This project was built to explore the LangChain ecosystem, stateful LLM applications, and the real-world constraints of deploying AI pipelines into cloud environments.

## 🚀 Live Demo
- **App:** [[text](https://youtube-rag-chatbot-usuallyhamza.streamlit.app/)]
- **Note:** The app currently supports English YouTube videos. Due to YouTube's strict datacenter IP firewalls, cloud deployments may occasionally experience scraping blocks. 

---

## ⚙️ Architecture & Tech Stack

The application strictly bounds the LLM's knowledge to the specific context of the provided video, eliminating hallucination.

* **Frontend UI:** Streamlit (Community Cloud)
* **Orchestration:** LangChain (using LCEL - LangChain Expression Language)
* **Document Ingestion:** `youtube-transcript-api`
* **Text Splitting:** `RecursiveCharacterTextSplitter` (Chunk size: 1000, Overlap: 200)
* **Embeddings:** Hugging Face `BAAI/bge-small-en-v1.5` (Local execution)
* **Vector Database:** FAISS (Facebook AI Similarity Search) CPU 
* **LLM:** Google Gemini (`gemini-1.5-flash`) via `ChatGoogleGenerativeAI`

---

## 🧠 Engineering Trade-offs & Challenges

Building this pipeline revealed several critical differences between local development and cloud production:

**1. Mitigating Cloud API Quotas (The 429 Error)**
Initial prototypes utilized cloud-based embedding endpoints. However, processing standard YouTube transcripts quickly exhausted free-tier rate limits, causing pipeline failures. 
* *Solution:* Decoupled the embedding architecture from cloud APIs by migrating to a local, open-source model (`BAAI/bge-small-en-v1.5`). This eliminated network latency, completely bypassed rate limits, and kept the compute footprint light enough to run on standard CPUs.

**2. Handling Ephemeral Infrastructure Constraints**
Streamlit Cloud's execution model re-runs scripts upon every user interaction. Without proper state management, the application would re-download the transcript, reload the 133MB embedding model, and rebuild the FAISS index on every chat message, leading to severe latency and Out-Of-Memory (OOM) crashes.
* *Solution:* Implemented `@st.cache_resource` decorators and Streamlit Session State to ensure the heavy ingestion and indexing pipeline executes only once per unique video URL.

**3. Navigating IP Firewalls**
While the `youtube-transcript-api` successfully extracts data locally, deploying to AWS/GCP-backed Streamlit containers introduced Datacenter IP blocking from YouTube's bot-detection algorithms.
* *Solution:* Explored proxy server integration to route traffic through residential IPs, highlighting the complexities of reliable data ingestion in production MLOps environments. 

---