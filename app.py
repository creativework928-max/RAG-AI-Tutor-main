import os
import json
import re
import requests
import streamlit as st

from PyPDF2 import PdfReader


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG AI Tutor",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "rag_data"

DOCUMENTS_FILE = os.path.join(
    DATA_DIR,
    "documents.json"
)

# ------------------------------------------------------------
# Ollama Cloud
# ------------------------------------------------------------

OLLAMA_API_URL = "https://ollama.com/api/chat"

# Cloud-enabled Qwen model.
#
# You can change this later to another Ollama Cloud model.
#
# IMPORTANT:
# Do NOT use qwen3:1.7b here.
# qwen3:1.7b is a local model.
#
OLLAMA_MODEL = "qwen3.5:2b"

# Maximum generated tokens
MAX_OUTPUT_TOKENS = 800

# Model temperature
TEMPERATURE = 0.3

# Number of retrieved chunks
TOP_K = 5

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# OLLAMA CLOUD API KEY
# ============================================================

def get_ollama_api_key():
    """
    Get Ollama Cloud API key from Streamlit Secrets.

    Expected Streamlit secret:

        OLLAMA_API_KEY = "your-api-key"
    """

    try:

        api_key = st.secrets.get(
            "OLLAMA_API_KEY",
            ""
        )

        return str(api_key).strip()

    except Exception:

        return ""


OLLAMA_API_KEY = get_ollama_api_key()


# ============================================================
# OLLAMA CLOUD CONNECTION TEST
# ============================================================

@st.cache_data(ttl=300)
def test_ollama_cloud():

    """
    Test whether Ollama Cloud credentials are configured.

    This does NOT install Ollama.
    This does NOT start a local Ollama server.

    It only verifies that an API key exists.
    """

    api_key = get_ollama_api_key()

    if not api_key:

        return False, (
            "OLLAMA_API_KEY is not configured."
        )

    return True, "Ollama Cloud API key configured."


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

IMPORTANT:
Do not expose your internal reasoning or thinking process.

Do not show <think></think> content in the answer.

Use ONLY the provided context to answer factual questions.

If the answer cannot be found in the provided context, clearly say:

"The information is not available in the uploaded documents."

Do not invent information.

Format the answer in exactly two major parts:

1. 2 Mark Answer

- Give the core definition/key point.
- Keep it short and exam-friendly.
- Around 4-5 lines.
- Use simple language.

2. 16 Mark Answer

- Start with a short introduction.
- Explain 5-6 important sub-topics/key points.
- Give suitable examples.
- Describe diagrams in words when useful.
- Use simple analogies where appropriate.
- End with a conclusion.
- Keep the answer structured and useful for exam preparation.
- Make it engaging like a Gen-Z teacher.

------------------------------------------------------------
PROVIDED CONTEXT
------------------------------------------------------------

{context}

------------------------------------------------------------
QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ANSWER
------------------------------------------------------------
"""


# ============================================================
# LOAD DOCUMENT DATABASE
# ============================================================

def load_documents():
    """
    Load saved documents from JSON.
    """

    if not os.path.exists(
        DOCUMENTS_FILE
    ):
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

        st.error(
            f"Could not load document database: {e}"
        )

    return []


# ============================================================
# SAVE DOCUMENT DATABASE
# ============================================================

def save_documents(documents):
    """
    Save documents to local JSON.
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

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if filename.endswith(".pdf"):

        try:

            reader = PdfReader(
                uploaded_file
            )

            pages = []

            for page in reader.pages:

                try:

                    text = page.extract_text()

                    if text:
                        pages.append(text)

                except Exception:
                    continue

            return "\n".join(pages)

        except Exception as e:

            st.error(
                f"Could not read PDF "
                f"'{uploaded_file.name}': {e}"
            )

            return None

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    if filename.endswith(".txt"):

        try:

            data = uploaded_file.read()

            return data.decode(
                "utf-8",
                errors="ignore"
            )

        except Exception as e:

            st.error(
                f"Could not read TXT "
                f"'{uploaded_file.name}': {e}"
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

    text = text.replace(
        "\x00",
        " "
    )

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
    Split document into overlapping word chunks.

    No vector database required.
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

            chunks.append(
                chunk
            )

        if end >= len(words):
            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


# ============================================================
# TOKENIZE
# ============================================================

def tokenize(text):
    """
    Convert text into lowercase word tokens.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


# ============================================================
# TEXT SIMILARITY
# ============================================================

def similarity_score(
    query,
    document
):
    """
    Simple Jaccard-style similarity.

    No NumPy.
    No embeddings.
    No vector database.
    """

    query_words = tokenize(
        query
    )

    document_words = tokenize(
        document
    )

    if (
        not query_words
        or not document_words
    ):
        return 0.0

    intersection = (
        query_words &
        document_words
    )

    union = (
        query_words |
        document_words
    )

    if not union:
        return 0.0

    score = (
        len(intersection)
        / len(union)
    )

    # --------------------------------------------------------
    # Exact phrase bonus
    # --------------------------------------------------------

    query_lower = (
        query.lower().strip()
    )

    document_lower = (
        document.lower()
    )

    if (
        query_lower
        and query_lower in document_lower
    ):

        score += 1.0

    # --------------------------------------------------------
    # Important word bonus
    # --------------------------------------------------------

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
    top_k=TOP_K
):
    """
    Retrieve the most relevant document chunks.
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
    Add a document to JSON knowledge base.
    """

    text = clean_text(
        text
    )

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

    # Remove old copy
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

    save_documents(
        documents
    )

    return len(chunks)


# ============================================================
# REMOVE THINKING TAGS
# ============================================================

def clean_model_response(response):
    """
    Remove accidental thinking tags from the model response.
    """

    if not response:
        return ""

    response = str(
        response
    )

    # Remove complete <think>...</think>
    response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL
        | re.IGNORECASE
    )

    # Remove orphan tags
    response = re.sub(
        r"</?think>",
        "",
        response,
        flags=re.IGNORECASE
    )

    return response.strip()


# ============================================================
# CALL OLLAMA CLOUD
# ============================================================

def call_ollama_cloud(
    question,
    context
):
    """
    Send the RAG prompt directly to Ollama Cloud.

    IMPORTANT:

    This function DOES NOT:

    - install Ollama
    - run Ollama locally
    - use localhost
    - use ollama serve
    - download a model

    It sends HTTPS requests directly to:

        https://ollama.com/api/chat
    """

    api_key = get_ollama_api_key()

    if not api_key:

        raise RuntimeError(
            "Ollama Cloud API key is missing. "
            "Add OLLAMA_API_KEY to Streamlit Secrets."
        )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    headers = {
        "Authorization": (
            f"Bearer {api_key}"
        ),
        "Content-Type": "application/json"
    }

    payload = {
        "model": OLLAMA_MODEL,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "stream": False,

        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_OUTPUT_TOKENS
        }
    }

    try:

        response = requests.post(
            OLLAMA_API_URL,
            headers=headers,
            json=payload,
            timeout=180
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "Ollama Cloud request timed out. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to Ollama Cloud. "
            "Please check your internet connection."
        )

    except requests.exceptions.RequestException as e:

        raise RuntimeError(
            f"Ollama Cloud connection error: {e}"
        )

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    if response.status_code in (
        401,
        403
    ):

        raise RuntimeError(
            "Ollama Cloud authentication failed. "
            "Your OLLAMA_API_KEY may be invalid or expired."
        )

    # --------------------------------------------------------
    # Rate limit
    # --------------------------------------------------------

    if response.status_code == 429:

        raise RuntimeError(
            "Ollama Cloud rate limit or usage limit "
            "was reached. Please try again later."
        )

    # --------------------------------------------------------
    # Other HTTP errors
    # --------------------------------------------------------

    if response.status_code != 200:

        try:

            error_data = (
                response.json()
            )

            error_message = (
                error_data.get(
                    "error",
                    response.text
                )
            )

        except Exception:

            error_message = (
                response.text
            )

        raise RuntimeError(
            f"Ollama Cloud returned "
            f"HTTP {response.status_code}: "
            f"{error_message}"
        )

    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            "Ollama Cloud returned an invalid response."
        )

    # Ollama chat API response:
    #
    # {
    #   "message": {
    #       "role": "assistant",
    #       "content": "..."
    #   }
    # }

    message = data.get(
        "message"
    )

    if not isinstance(
        message,
        dict
    ):

        raise RuntimeError(
            "Ollama Cloud response did not "
            "contain an assistant message."
        )

    answer = message.get(
        "content",
        ""
    )

    if not answer:

        raise RuntimeError(
            "Ollama Cloud returned an empty answer."
        )

    return clean_model_response(
        answer
    )


# ============================================================
# RAG QUERY
# ============================================================

def query_rag(question):
    """
    Complete RAG pipeline:

    1. Load JSON documents
    2. Retrieve relevant chunks
    3. Build context
    4. Send context to Ollama Cloud
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
        top_k=TOP_K
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

            sources.append(
                source
            )

    context = (
        "\n\n---\n\n".join(
            context_parts
        )
    )

    answer = call_ollama_cloud(
        question,
        context
    )

    return (
        answer,
        sources
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📚 RAG-Chatbot"
)

st.markdown(
    "Chat with your PDFs like a "
    "**Gen-Z Teacher** 📚"
)


# ============================================================
# OLLAMA CLOUD STATUS
# ============================================================

cloud_ready, cloud_message = (
    test_ollama_cloud()
)

if cloud_ready:

    st.success(
        f"☁️ Ollama Cloud connected • "
        f"`{OLLAMA_MODEL}`"
    )

else:

    st.warning(
        "☁️ Ollama Cloud API key is not configured yet."
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

st.subheader(
    "📂 Upload New Docs"
)

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

            st.rerun()

        else:

            st.error(
                "No readable text was found "
                "in the uploaded documents. ❌"
            )


# ============================================================
# ASK QUESTIONS
# ============================================================

st.subheader(
    "💬 Ask Questions"
)

query_text = st.text_input(
    "Ask me anything from your docs:",
    placeholder=(
        "Example: What is Human Values?"
    )
)


if st.button(
    "Ask"
):

    if not query_text.strip():

        st.warning(
            "Please type a question first. 🤌"
        )

    elif not cloud_ready:

        st.error(
            "☁️ Ollama Cloud is not configured. "
            "Please add your OLLAMA_API_KEY "
            "to Streamlit Secrets."
        )

    else:

        try:

            with st.spinner(
                "Searching your documents and "
                "asking Ollama Cloud... 🦙"
            ):

                response, sources = (
                    query_rag(
                        query_text
                    )
                )

            st.subheader(
                "📝 Answer"
            )

            st.write(
                response
            )

            if sources:

                st.subheader(
                    "📌 Sources"
                )

                for source in sources:

                    st.code(
                        source,
                        language="text"
                    )

        except Exception as e:

            st.error(
                "The question could not be processed. ❌"
            )

            st.error(
                str(e)
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Settings"
    )

    st.write(
        f"**AI:** Ollama Cloud"
    )

    st.write(
        f"**Model:** `{OLLAMA_MODEL}`"
    )

    st.write(
        "**Storage:** Local JSON"
    )

    st.write(
        "**Vector database:** Not required"
    )

    st.divider()

    if cloud_ready:

        st.success(
            "☁️ Ollama Cloud: Connected"
        )

    else:

        st.warning(
            "☁️ Ollama Cloud: Not configured"
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
