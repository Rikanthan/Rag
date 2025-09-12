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
# Vector stores
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

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


# --- Chroma ---
def get_new_chroma_path(base_path="./chroma_db"):
    i = 1
    while os.path.exists(f"{base_path}{i}"):
        i += 1
    return f"{base_path}{i}"


def file_already_indexed_chroma(vectorstore, file_hash):
    try:
        results = vectorstore._collection.get(where={"file_hash": file_hash})
        return len(results["ids"]) > 0
    except Exception:
        return False


# --- Supabase ---
def get_new_supabase_table(client, base_name="document"):
    """Find next available Supabase table like document1, document2..."""
    i = 1
    while True:
        table_name = f"{base_name}{i}"
        try:
            client.table(table_name).select("id").limit(1).execute()
            i += 1
        except Exception:
            return table_name


def create_supabase_table_and_function(client, table_name):
    """Create table + match function for Supabase"""
    ddl = f"""
    create table if not exists {table_name} (
        id uuid primary key default gen_random_uuid(),
        content text,
        metadata jsonb,
        embedding vector(768)
    );
    """
    fn = f"""
    create or replace function match_{table_name}(
        query_embedding vector(768),
        match_count int DEFAULT 5
    ) returns table (
        id uuid,
        content text,
        metadata jsonb,
        similarity float
    )
    language sql stable as $$
        select
            id,
            content,
            metadata,
            1 - (embedding <=> query_embedding) as similarity
        from {table_name}
        order by embedding <=> query_embedding
        limit match_count;
    $$;
    """
    client.rpc("execute_sql", {"sql": ddl}).execute()
    client.rpc("execute_sql", {"sql": fn}).execute()
    st.success(f"✅ Supabase table + function ready: {table_name}, match_{table_name}")


def file_already_indexed_supabase(client, table_name, file_hash):
    try:
        res = (
            client.table(table_name)
            .select("id")
            .eq("metadata->>file_hash", file_hash)
            .limit(1)
            .execute()
        )
        return len(res.data) > 0
    except Exception:
        return False


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
    if ENVIRONMENT == "local":
        db_path = get_new_chroma_path()
        st.info(f"Using Chroma vector store (local) → {db_path}")
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

        if file_already_indexed_chroma(vectorstore, file_hash):
            st.warning("⚠️ File already indexed in Chroma, skipping insert.")
        else:
            batch_size = 20
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                with st.spinner(f"🔄 Embedding batch {i//batch_size + 1}..."):
                    try:
                        safe_add_docs(vectorstore, batch)
                    except Exception as e:
                        st.error(f"Batch {i//batch_size + 1} failed in {db_path}: {e}")
                        db_path = get_new_chroma_path()
                        st.warning(f"➡️ Switching to new DB path: {db_path}")
                        vectorstore = Chroma(
                            persist_directory=db_path, embedding_function=embeddings
                        )
                        safe_add_docs(vectorstore, batch)
                    time.sleep(2)
            vectorstore.persist()
            st.success(f"✅ Documents stored in Chroma DB at {db_path}")

    else:
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
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2,
    )
    retriever = vectorstore.as_retriever()

    # Step 7: Build QA chain
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use only the following context to answer "
        "the question. "
        "If the context does not contain the answer, respond with: 'Sorry, question is not related to the document'. "
        "Limit your answer to no more than 10 words.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
    )

    # Step 8: Ask questions
    query = st.text_input("🔎 Ask a question:")
    if query:
        with st.spinner("Processing..."):
            result = rag_chain.run(query)
            st.write("### 📌 Result:", result)

else:
    st.info("⬆️ Please upload a PDF to start.")
