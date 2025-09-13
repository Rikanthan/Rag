import asyncio
import os
import tempfile
import time
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_google_genai.common import GoogleGenerativeAIError
from langchain_community.vectorstores import SupabaseVectorStore


from utils.db_functions import get_new_supabase_table, create_supabase_table_and_function,file_already_indexed_supabase
from tenacity import retry, wait_exponential, stop_after_attempt
from google.api_core.exceptions import ResourceExhausted, InvalidArgument

# --- Config ---
load_dotenv()
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Streamlit UI ---
st.title("📚 RAG App: Chroma (local) / Supabase (prod)")
uploadedfile = st.file_uploader("Upload a PDF file", type=["pdf"])


# --- Helpers ---
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def compute_file_hash(file_path):
    """Return MD5 hash of a file"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(6))
def safe_add_docs(vectorstore, docs):
    try:
        vectorstore.add_documents(docs)
    except GoogleGenerativeAIError as e:
        st.error(f"❌ Gemini error: {e}")
        raise
    except ResourceExhausted as e:
        st.warning(f"⚠️ Rate limit exceeded, retrying: {e}")
        raise
    except InvalidArgument as e:
        st.warning(f"⚠️ Chunk too big or bad input: {e}")
        raise
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
        raise


def init_embeddings():
    ensure_event_loop()
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# --- Main Flow ---
if uploadedfile:
    # Step 1: Load PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploadedfile.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    st.success(f"✅ Loaded {len(docs)} pages")

    # Step 2: Split docs
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    st.info(f"✅ Split into {len(chunks)} chunks")

    # Step 3: Compute file hash
    file_hash = compute_file_hash(tmp_path)
    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash

    # Step 4: Init embeddings
    embeddings = init_embeddings()

    # Step 5: Store embeddings
    st.info("Using Supabase vector store (production)")
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # New table + function
    table_name = get_new_supabase_table(supabase_client, "document")
    create_supabase_table_and_function(supabase_client, table_name)

    if file_already_indexed_supabase(supabase_client, table_name, file_hash):
        st.warning(f"⚠️ File already indexed in {table_name}, skipping insert.")
    else:
        vectorstore = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name=table_name,
            query_name=f"match_{table_name}",
        )

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            with st.spinner(f"🔄 Uploading batch {i//batch_size + 1} to Supabase..."):
                try:
                    safe_add_docs(vectorstore, batch)
                except Exception as e:
                    st.error(f"❌ Supabase batch {i//batch_size + 1} failed: {e}")
                time.sleep(2)

        st.success(f"✅ Documents stored in Supabase table '{table_name}'")

    # Step 6: Init LLM
else:
    st.info("⬆️ Please upload a PDF to start.")
