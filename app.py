import os
import json
import math
import re
import streamlit as st

from PyPDF2 import PdfReader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "rag_data"
DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.json")

OLLAMA_MODEL = "qwen3:1.7b"

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are a fun, Gen-Z style teacher who explains concepts in a
chill, engaging, and easy-to-understand way.

Your student is preparing for exams, so be precise and give
correct information from the provided context.

Your goal is to make students understand the concept clearly,
like explaining it to a friend.

Add a few appropriate emojis and make the content fun to read.

Do not show <think></think> content in the answer.

Format the answer in two parts:

1. 2 Mark Answer
   - Give the core definition/key point.
   - Keep it short and exam-friendly.
   - Around 4-5 lines.

2. 16 Mark Answer
   - Give a detailed explanation.
   - Start with an introduction.
   - Explain 5-6 important sub-topics/key points.
   - Give suitable examples.
   - Describe diagrams in words when useful.
   - Use simple analogies where appropriate.
   - End with a conclusion.
   - Keep the answer structured and useful for exam preparation.
   - Make it engaging like a Gen-Z teacher.

IMPORTANT:
Use ONLY the provided context to answer factual questions.

If the answer cannot be found in the context, clearly say:

"The information is not available in the uploaded documents."

Do not invent information.

Context:
{context}

Question:
{question}

Answer in the format described above:
"""


# ============================================================
# LOAD DOCUMENT DATABASE
# ============================================================

def load_documents():
    """
    Load saved documents from JSON.
    """

    if not os.path.exists(DOCUMENTS_FILE):
        return []

    try:
        with open(
            DOCUMENTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            if isinstance(data, list):
                return data

    except Exception as e:
        st.error(f"Could not load document database: {e}")

    return []


# ============================================================
# SAVE DOCUMENT DATABASE
# ============================================================

def save_documents(documents):
    """
    Save documents to JSON.
    """

    with open(
        DOCUMENTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_from_file(uploaded_file):
    """
    Extract text from PDF or TXT.
    """

    filename = uploaded_file.name.lower()

    # -----------------------------
    # PDF
    # -----------------------------

    if filename.endswith(".pdf"):

        try:

            reader = PdfReader(uploaded_file)

            pages = []

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    pages.append(text)

            return "\n".join(pages)

        except Exception as e:

            st.error(
                f"Could not read PDF '{uploaded_file.name}': {e}"
            )

            return None

    # -----------------------------
    # TXT
    # -----------------------------

    if filename.endswith(".txt"):

        try:

            data = uploaded_file.read()

            return data.decode(
                "utf-8",
                errors="ignore"
            )

        except Exception as e:

            st.error(
                f"Could not read TXT '{uploaded_file.name}': {e}"
            )

            return None

    return None


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean extracted document text.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SPLIT DOCUMENT INTO CHUNKS
# ============================================================

def split_into_chunks(
    text,
    chunk_size=1200,
    overlap=200
):
    """
    Split document into smaller chunks.

    This implementation uses only Python's standard library.
    No NumPy, ChromaDB, or native DLLs are required.
    """

    text = clean_text(text)

    if not text:
        return []

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


# ============================================================
# TOKENIZE TEXT
# ============================================================

def tokenize(text):
    """
    Convert text into simple lowercase word tokens.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


# ============================================================
# SIMPLE TEXT SIMILARITY
# ============================================================

def similarity_score(
    query,
    document
):
    """
    Calculate simple similarity using word overlap.

    This deliberately avoids NumPy and native vector libraries.
    """

    query_words = tokenize(query)
    document_words = tokenize(document)

    if not query_words or not document_words:
        return 0.0

    intersection = (
        query_words & document_words
    )

    # Basic Jaccard similarity
    union = (
        query_words | document_words
    )

    if not union:
        return 0.0

    score = (
        len(intersection)
        / len(union)
    )

    # Extra score for exact phrase matches
    query_lower = query.lower().strip()
    document_lower = document.lower()

    if (
        query_lower
        and query_lower in document_lower
    ):
        score += 1.0

    # Extra score for important individual words
    important_words = [
        word
        for word in query_words
        if len(word) >= 4
    ]

    if important_words:

        matched = sum(
            1
            for word in important_words
            if word in document_words
        )

        score += (
            matched
            / len(important_words)
        ) * 0.5

    return score


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    query,
    documents,
    top_k=5
):
    """
    Find the most relevant chunks.
    """

    scored_chunks = []

    for document in documents:

        source = document.get(
            "source",
            "Unknown"
        )

        chunks = document.get(
            "chunks",
            []
        )

        for chunk in chunks:

            score = similarity_score(
                query,
                chunk
            )

            if score > 0:

                scored_chunks.append(
                    {
                        "score": score,
                        "text": chunk,
                        "source": source
                    }
                )

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_chunks[:top_k]


# ============================================================
# ADD DOCUMENT
# ============================================================

def add_document(
    filename,
    text
):
    """
    Add a document to the local JSON knowledge base.
    """

    text = clean_text(text)

    if not text:
        return 0

    chunks = split_into_chunks(
        text,
        chunk_size=1200,
        overlap=200
    )

    if not chunks:
        return 0

    documents = load_documents()

    # Remove previous copy of same filename
    documents = [
        document
        for document in documents
        if document.get("source") != filename
    ]

    documents.append(
        {
            "source": filename,
            "chunks": chunks
        }
    )

    save_documents(documents)

    return len(chunks)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context
):
    """
    Send retrieved context to Ollama.
    """

    prompt_template = (
        ChatPromptTemplate.from_template(
            PROMPT_TEMPLATE
        )
    )

    prompt = prompt_template.format(
        context=context,
        question=question
    )

    model = OllamaLLM(
        model=OLLAMA_MODEL,
        options={
            "num_predict": 800,
            "temperature": 0.3
        }
    )

    response = model.invoke(prompt)

    if response is None:
        return "No response was generated."

    response = str(response)

    # Remove <think>...</think>
    response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL
    )

    return response.strip()


# ============================================================
# RAG QUERY
# ============================================================

def query_rag(question):
    """
    Retrieve relevant chunks and ask Ollama.
    """

    documents = load_documents()

    if not documents:

        return (
            "No documents have been added yet. "
            "Please upload a PDF or TXT file first.",
            []
        )

    results = retrieve_relevant_chunks(
        question,
        documents,
        top_k=5
    )

    if not results:

        return (
            "I couldn't find relevant information "
            "in the uploaded documents. ❌",
            []
        )

    context_parts = []

    sources = []

    for result in results:

        context_parts.append(
            result["text"]
        )

        source = result["source"]

        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(
        context_parts
    )

    response = generate_answer(
        question,
        context
    )

    return response, sources


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG AI Tutor",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title("📚 RAG-Chatbot")

st.markdown(
    "Chat with your PDFs like a **Gen-Z Teacher** 📚"
)


# ============================================================
# DATABASE STATUS
# ============================================================

documents = load_documents()

if documents:

    total_chunks = sum(
        len(
            document.get(
                "chunks",
                []
            )
        )
        for document in documents
    )

    st.info(
        f"📚 {len(documents)} document(s) loaded "
        f"• {total_chunks} text chunks"
    )

else:

    st.info(
        "📭 No documents added yet."
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader("📂 Upload New Docs")

uploaded_files = st.file_uploader(
    "Upload your notes (PDF or TXT)",
    type=[
        "pdf",
        "txt"
    ],
    accept_multiple_files=True
)


if uploaded_files:

    if st.button(
        "➕ Add to Knowledge Base"
    ):

        total_chunks_added = 0

        with st.spinner(
            "Reading your documents... 📖"
        ):

            for uploaded_file in uploaded_files:

                text = extract_text_from_file(
                    uploaded_file
                )

                if text:

                    chunk_count = add_document(
                        uploaded_file.name,
                        text
                    )

                    total_chunks_added += (
                        chunk_count
                    )

        if total_chunks_added > 0:

            st.success(
                f"Documents added successfully! ✅ "
                f"{total_chunks_added} text chunks stored."
            )

        else:

            st.error(
                "No readable text was found "
                "in the uploaded documents. ❌"
            )


# ============================================================
# ASK QUESTIONS
# ============================================================

st.subheader("💬 Ask Questions")

query_text = st.text_input(
    "Ask me anything from your docs:",
    placeholder="Example: What is Human Values?"
)


if st.button("Ask"):

    if not query_text.strip():

        st.warning(
            "Please type a question first. 🤌"
        )

    else:

        try:

            with st.spinner(
                "Searching your documents... 🔎"
            ):

                response, sources = query_rag(
                    query_text
                )

            st.subheader("📝 Answer")

            st.write(response)

            if sources:

                st.subheader("📌 Sources")

                for source in sources:

                    st.code(
                        source,
                        language="text"
                    )

        except Exception as e:

            st.error(
                "The question could not be processed."
            )

            st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        f"**Ollama model:** `{OLLAMA_MODEL}`"
    )

    st.write(
        "**Storage:** Local JSON"
    )

    st.write(
        "**Vector database:** Not required"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Knowledge Base"
    ):

        if os.path.exists(
            DOCUMENTS_FILE
        ):

            os.remove(
                DOCUMENTS_FILE
            )

        st.success(
            "Knowledge base cleared! ✅"
        )

        st.rerun()