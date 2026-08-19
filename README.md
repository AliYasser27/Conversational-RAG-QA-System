# Conversational RAG Q&A with PDF Upload and Chat History

A professional, object-oriented Conversational Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, and **Chroma DB**. This application allows users to upload multiple PDF documents, process and index them into a vector database, and engage in context-aware, multi-session dialogues powered by Groq's high-performance Large Language Models.

---

## 🚀 Key Features

- **Multi-Session Chat Management:** Create, switch between, and delete isolated chat sessions. Each session maintains its own independent conversation memory and state.
- **Robust PDF Ingestion Pipeline:** Dynamically processes multiple PDFs at runtime, split into optimal chunks via `RecursiveCharacterTextSplitter`, embedded locally using HuggingFace models, and indexed in Chroma DB.
- **Object-Oriented Architecture (Strategy Pattern):** Promotes loose coupling by abstracting the chat engine mechanism into a flexible `ChatEngine` protocol. Easily swap between the legacy `SimpleFAQEngine` and the advanced `RagEngine`.
- **History-Aware Question Reformulation:** Uses a specialized two-step LangChain LCEL pipeline to re-contextualize ambiguous follow-up questions into standalone queries based on preceding chat history before performing vector retrieval.
- **Secure Runtime Configuration:** Externalizes hyperparameters (chunk size, overlap, model selection, retrieval `k`) into a clean configuration layer while capturing API keys securely via UI input fields (avoiding plaintext disk storage).

---

## 📂 Directory Structure

```text
├── app.py                 # Main Streamlit UI and session orchestration layer
├── config.yaml            # Central configuration file for models, paths, and tunables
├── requirements.txt       # Project software dependencies
├── README.md              # Project documentation and setup instructions
├── engines/
│   ├── __init__.py
│   ├── base.py            # ChatEngine Protocol (interface definition)
│   ├── simple_faq.py      # Legacy/Fallback Simple FAQ engine
│   └── rag_engine.py      # Main LangChain-backed RAG engine implementation
└── rag/
    ├── __init__.py
    ├── ingest.py          # Document extraction, text chunking, and embedding storage
    └── pipeline.py        # LCEL chain configuration and history management
```

## 🛠️ Prerequisites & Configuration

### 1. config.yaml

Ensure you have a `config.yaml` file in your root directory matching the parameters used by the codebase:

```yaml
chunking:
  chunk_size: 1000
  chunk_overlap: 200

embeddings:
  model: "all-MiniLM-L6-v2"

vector_store:
  collection_name: "student-rag"
  persist_directory: ".chroma/student-rag"

llm:
  model: "llama3-8b-8192"
  temperature: 0.3

retrieval:
  k: 3
```

## 💻 Installation & Setup

1. **Clone or Extract the Project:**  
   Ensure all source files are organized according to the directory layout shown above.

2. **Create a Virtual Environment:**
   ```bash
   # Create environment
   python -m venv venv

   # Activate environment (Windows)
   venv\Scripts\activate

   # Activate environment (Linux/macOS)
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Your `requirements.txt` should include: `streamlit`, `langchain`, `langchain-community`, `langchain-core`, `langchain-groq`, `langchain-huggingface`, `langchain-chroma`, `pypdf`, `pyyaml`, `sentence-transformers`.

## 🏃 How to Run the Application

Start the Streamlit application server by running:

```bash
streamlit run app.py
```

Once running, navigate to the local URL provided in your terminal (typically `http://localhost:8501`).

## 📖 Step-by-Step Usage Guide

1. **Add Your API Key:**  
   Obtain an API key from the [Groq Console](https://console.groq.com) and input it into the secure sidebar text box.

2. **Upload PDF Documents:**  
   Drag and drop or browse for the PDFs you wish to chat with under the **Upload PDFs** section.

3. **Process Ingestion:**  
   Click **🚀 Process PDFs**. The app will extract text, chunk it, generate local embeddings, and populate your Chroma vector store.

4. **Interact and Converse:**
   - Type your question into the chat box.
   - For follow-up questions (e.g., asking *"Can you summarize its main points?"* after discussing a specific section), the pipeline automatically reads the session's chat history to rewrite your question into an explicit, searchable query.

5. **Manage Multiple Chats:**  
   Use the **➕ New Chat** button in the sidebar to spawn independent conversations. Switch back and forth seamlessly without cross-contamination or loss of state.
