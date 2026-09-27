
import os
import time
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

st.title("📚 PDF RAG Chatbot")
st.caption(
    "Upload a PDF, retrieve relevant information, "
    "and ask questions about your document."
)


# --------------------------------------------------
# 2. Initialize session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None


# --------------------------------------------------
# 3. Load models
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_gemini():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0
    )


# --------------------------------------------------
# 4. PDF processing function
# --------------------------------------------------

def process_pdf(uploaded_file):
    """
    Save the uploaded PDF temporarily, extract its text,
    split into chunks, generate embeddings, and create
    a fresh Chroma collection.
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    try:
        # Extract PDF text
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No pages were found in this PDF.")

        # Split extracted text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No readable text was extracted. "
                "The PDF may be scanned or image-only."
            )

        # Load local HuggingFace embeddings
        embedding_model = load_embedding_model()

        # Create a separate Chroma collection
        # for this upload.
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=f"pdf_{uuid.uuid4().hex}",
            # No persist_directory: in-memory collection.
        )

        return vectorstore, len(documents), len(chunks)

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# --------------------------------------------------
# 5. Sidebar: PDF upload and indexing
# --------------------------------------------------

with st.sidebar:
    st.header("📄 Your document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Choose a PDF file to create a searchable index."
    )

    if uploaded_file is not None:
        st.info(f"Selected: {uploaded_file.name}")

        if st.button(
            "Process PDF",
            type="primary",
            use_container_width=True
        ):
            # Clear the old document and chat when
            # a new indexing operation starts.
            st.session_state.vectorstore = None
            st.session_state.filename = None
            st.session_state.messages = []

            try:
                with st.spinner(
                    "Extracting text and creating embeddings..."
                ):
                    vectorstore, page_count, chunk_count = (
                        process_pdf(uploaded_file)
                    )

                st.session_state.vectorstore = vectorstore
                st.session_state.filename = uploaded_file.name

                st.success("PDF indexed successfully!")
                st.write(f"**Pages:** {page_count}")
                st.write(f"**Chunks:** {chunk_count}")

            except Exception as e:
                st.error(f"Could not process PDF: {e}")

    if st.session_state.vectorstore is not None:
        st.divider()
        st.success("Document ready for questions")
        st.write(f"**Current PDF:** {st.session_state.filename}")

        if st.button(
            "Clear conversation",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.rerun()

        if st.button(
            "Remove document",
            use_container_width=True
        ):
            st.session_state.vectorstore = None
            st.session_state.filename = None
            st.session_state.messages = []
            st.rerun()


# --------------------------------------------------
# 6. Create the RAG prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant that answers
        questions about an uploaded PDF document.

        Use ONLY the retrieved context below to answer.

        Instructions:
        1. Do not use outside knowledge.
        2. Give clear, accurate, well-organized answers.
        3. If the context does not contain the answer,
           say: "I could not find the answer in the document."
        4. If only part of the answer is supported,
           explain which part is supported.
        5. Do not invent facts or references.
        6. When possible, mention the page numbers
           included in the retrieved context.

        Retrieved context:
        {context}
        """
    ),
    (
        "human",
        "{question}"
    )
])


# --------------------------------------------------
# 7. Gemini retry function
# --------------------------------------------------

def invoke_with_retry(llm, messages, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(messages)

        except Exception as e:
            error_message = str(e).upper()

            temporary_error = (
                "503" in error_message
                or "UNAVAILABLE" in error_message
                or "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            )

            if not temporary_error or attempt == max_attempts:
                raise

            time.sleep(2 ** attempt)


# --------------------------------------------------
# 8. Retrieval and question-answering function
# --------------------------------------------------

def get_answer(question):
    vectorstore = st.session_state.vectorstore

    # MMR retrieval, matching the original CLI app
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # Retrieve relevant document chunks
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return (
            "I could not find the answer in the document.",
            []
        )

    # Build context and retain source page metadata
    context_parts = []

    for doc in retrieved_docs:
        page_number = doc.metadata.get("page", None)

        if page_number is not None:
            source_page = int(page_number) + 1
            context_parts.append(
                f"[Source: page {source_page}]\n"
                f"{doc.page_content}"
            )
        else:
            context_parts.append(doc.page_content)

    context = "\n\n".join(context_parts)

    # Generate response using Gemini
    llm = load_gemini()

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = invoke_with_retry(
        llm,
        final_prompt,
        max_attempts=3
    )

    answer = response.content

    # Normalize response to text
    if isinstance(answer, list):
        answer = "\n".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    return str(answer), retrieved_docs


# --------------------------------------------------
# 9. Main chat interface
# --------------------------------------------------

if st.session_state.vectorstore is None:
    st.info(
        "👈 Upload a PDF in the sidebar and click "
        "**Process PDF** to start asking questions."
    )

    st.markdown(
        """
        ### What you can do here

        - Ask questions about your PDF.
        - Retrieve relevant passages from the document.
        - See the source pages used to generate an answer.
        """
    )

else:
    st.subheader(f"Chatting with: {st.session_state.filename}")

    # Show conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("View retrieved sources"):
                        for i, doc in enumerate(sources, start=1):
                            page = doc.metadata.get("page")

                            if page is not None:
                                st.markdown(
                                    f"**Source {i} — Page {int(page) + 1}**"
                                )
                            else:
                                st.markdown(f"**Source {i}**")

                            st.write(doc.page_content)
                            st.divider()

    # Accept a new question
    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your PDF and generating answer..."):
                try:
                    answer, retrieved_docs = get_answer(question)

                    st.markdown(answer)

                    if retrieved_docs:
                        with st.expander("View retrieved sources"):
                            for i, doc in enumerate(
                                retrieved_docs, start=1
                            ):
                                page = doc.metadata.get("page")

                                if page is not None:
                                    st.markdown(
                                        f"**Source {i} — Page "
                                        f"{int(page) + 1}**"
                                    )
                                else:
                                    st.markdown(f"**Source {i}**")

                                st.write(doc.page_content)
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_docs
                    })

                except Exception as e:
                    error_message = str(e).upper()

                    if (
                        "429" in error_message
                        or "RESOURCE_EXHAUSTED" in error_message
                    ):
                        message = (
                            "Gemini API quota exceeded. "
                            "Check your API usage and try again later."
                        )
                    elif (
                        "503" in error_message
                        or "UNAVAILABLE" in error_message
                    ):
                        message = (
                            "Gemini is temporarily unavailable. "
                            "Please try again later."
                        )
                    else:
                        message = f"An error occurred: {e}"

                    st.error(message)