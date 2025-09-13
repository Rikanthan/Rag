import os
import tempfile
import time
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore

from utils.db_functions import file_already_indexed_supabase

# --- Config ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Streamlit UI ---
st.title("📚 RAG App: Supabase Vector Store")
uploadedfile = st.file_uploader("Upload a PDF file", type=["pdf"])


# --- Helpers ---
def compute_file_hash(file_path):
    """Return MD5 hash of a file"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def init_embeddings():
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

    # Step 5: Supabase storage (pre-created "documents" table)
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    table_name = "documents"

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
                    vectorstore.add_documents(batch)
                except Exception as e:
                    st.error(f"❌ Supabase batch {i//batch_size + 1} failed: {e}")
                time.sleep(2)

        st.success(f"✅ Documents stored in Supabase table '{table_name}'")
else:
    st.info("⬆️ Please upload a PDF to start.")
