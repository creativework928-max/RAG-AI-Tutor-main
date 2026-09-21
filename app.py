import os
import json
import re
import shutil
import subprocess
import time
import urllib.request
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
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# OLLAMA SETUP
# ============================================================

def run_command(command):
    """
    Run a shell command and return its output.
    """

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )

        return (
            result.returncode,
            result.stdout,
            result.stderr
        )

    except subprocess.TimeoutExpired:
        return (
            -1,
            "",
            "Command timed out."
        )

    except Exception as e:
        return (
            -1,
            "",
            str(e)
        )


def ollama_is_installed():
    """
    Check whether Ollama is installed.
    """

    return shutil.which("ollama") is not None


def install_ollama():
    """
    Install Ollama automatically if it is not installed.

    Ollama provides an official Linux installation script.
    """

    st.info("🦙 Ollama is not installed. Installing Ollama...")

    try:

        result = subprocess.run(
            [
                "bash",
                "-c",
                "curl -fsSL https://ollama.com/install.sh | sh"
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:

            st.error(
                "❌ Ollama installation failed."
            )

            st.code(
                result.stderr,
                language="text"
            )

            return False

        st.success(
            "✅ Ollama installed successfully."
        )

        return True

    except Exception as e:

        st.error(
            f"❌ Could not install Ollama: {e}"
        )

        return False


def ollama_server_running():
    """
    Check whether the Ollama server is responding.
    """

    try:

        with urllib.request.urlopen(
            f"{OLLAMA_URL}/api/tags",
            timeout=3
        ) as response:

            return response.status == 200

    except Exception:
        return False


def start_ollama_server():
    """
    Start Ollama in the background.
    """

    if ollama_server_running():
        return True

    st.info("🦙 Starting Ollama server...")

    try:

        env = os.environ.copy()

        env["OLLAMA_HOST"] = (
            f"{OLLAMA_HOST}:{OLLAMA_PORT}"
        )

        # Start Ollama in background.
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env
        )

    except Exception as e:

        st.error(
            f"❌ Could not start Ollama: {e}"
        )

        return False

    # Wait for server to become available.
    for _ in range(30):

        if ollama_server_running():

            st.success(
                "✅ Ollama server is running."
            )

            return True

        time.sleep(1)

    st.error(
        "❌ Ollama server did not start within "
        "the expected time."
    )

    return False


def ollama_model_exists():
    """
    Check whether the required model exists.
    """

    try:

        result = subprocess.run(
            [
                "ollama",
                "list"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return False

        return OLLAMA_MODEL in result.stdout

    except Exception:
        return False


def pull_ollama_model():
    """
    Download the required Ollama model.
    """

    st.info(
        f"📥 Downloading `{OLLAMA_MODEL}`..."
    )

    st.warning(
        "⚠️ The first startup may take several minutes "
        "because the model must be downloaded."
    )

    try:

        process = subprocess.Popen(
            [
                "ollama",
                "pull",
                OLLAMA_MODEL
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []

        if process.stdout:

            for line in process.stdout:

                line = line.strip()

                if line:
                    output_lines.append(line)

        return_code = process.wait()

        if return_code != 0:

            st.error(
                "❌ Failed to download the Ollama model."
            )

            if output_lines:

                st.code(
                    "\n".join(output_lines[-30:]),
                    language="text"
                )

            return False

        st.success(
            f"✅ `{OLLAMA_MODEL}` downloaded successfully."
        )

        return True

    except Exception as e:

        st.error(
            f"❌ Could not download model: {e}"
        )

        return False


@st.cache_resource
def initialize_ollama():
    """
    Complete Ollama initialization.

    This function:

    1. Checks whether Ollama is installed.
    2. Installs Ollama if necessary.
    3. Starts the Ollama server.
    4. Downloads qwen3:1.7b if necessary.
    """

    # --------------------------------------------------------
    # STEP 1 — Install Ollama
    # --------------------------------------------------------

    if not ollama_is_installed():

        if not install_ollama():

            return False

    # --------------------------------------------------------
    # STEP 2 — Start Ollama
    # --------------------------------------------------------

    if not start_ollama_server():

        return False

    # --------------------------------------------------------
    # STEP 3 — Download model
    # --------------------------------------------------------

    if not ollama_model_exists():

        if not pull_ollama_model():

            return False

    return True


# ============================================================
# INITIALIZE OLLAMA
# ============================================================

ollama_ready = initialize_ollama()


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

        st.error(
            f"Could not load document database: {e}"
        )

    return []


# ============================================================
# SAVE DOCUMENT DATABASE
# ============================================================

def save_documents(documents):

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

    filename = uploaded_file.name.lower()

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
                f"Could not read PDF "
                f"'{uploaded_file.name}': {e}"
            )

            return None

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

    query_words = tokenize(query)

    document_words = tokenize(document)

    if not query_words or not document_words:
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

    query_lower = query.lower().strip()

    document_lower = document.lower()

    if (
        query_lower
        and query_lower in document_lower
    ):
        score += 1.0

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

    if not ollama_ready:

        return (
            "❌ Ollama is not available. "
            "The AI model could not be started."
        )

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
        base_url=OLLAMA_URL,
        options={
            "num_predict": 800,
            "temperature": 0.3
        }
    )

    response = model.invoke(prompt)

    if response is None:
        return "No response was generated."

    response = str(response)

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
# OLLAMA STATUS
# ============================================================

if ollama_ready:

    st.success(
        f"🦙 Ollama is ready • `{OLLAMA_MODEL}`"
    )

else:

    st.error(
        "❌ Ollama could not be initialized."
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
