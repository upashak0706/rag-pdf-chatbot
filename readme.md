
# PDF RAG Chatbot

An AI-powered PDF question-answering application built with
Streamlit, LangChain, HuggingFace embeddings, ChromaDB,
and Google Gemini.

## Features

- Upload PDF documents through a web interface.
- Extract and split PDF text into searchable chunks.
- Generate embeddings locally using HuggingFace.
- Store and retrieve relevant passages using ChromaDB.
- Ask questions about uploaded documents using Gemini.
- View retrieved source passages and page numbers.
- Chat interface with conversation history.

## Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Sentence Transformers
- ChromaDB
- Google Gemini API
- PyPDF

## Project Setup

### 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/rag-pdf-chatbot.git

cd rag-pdf-chatbot

### 2. Create a virtual environment

python -m venv .venv

### 3. Activate the environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

### 4. Install dependencies

pip install -r requirements.txt

### 5. Configure your API key

Create a .env file in the root directory:

GOOGLE_API_KEY=your_api_key

### 6. Run the application

streamlit run app.py

## How It Works

1. The user uploads a PDF.
2. PyPDFLoader extracts its text.
3. Text is divided into overlapping chunks.
4. HuggingFace generates embeddings locally.
5. ChromaDB indexes the chunks for similarity retrieval.
6. The retriever selects relevant passages for each query.
7. Gemini generates an answer based on the retrieved context.

## Future Improvements

- Multiple PDF uploads
- OCR support for scanned PDFs
- Downloadable chat history
- Document-specific citations
- Persistent document storage