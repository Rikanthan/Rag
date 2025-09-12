import asyncio
import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# Vector stores
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

from tenacity import retry, wait_exponential, stop_after_attempt
from google.api_core.exceptions import ResourceExhausted, InvalidArgument

# --- Config ---
load_dotenv()
ENVIRONMENT = os.getenv("ENVIRONMENT")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Streamlit UI ---
st.title("📚 RAG App: Chroma (local) / Supabase (prod)")
uploadedfile = st.file_uploader("Upload a PDF file", type=["pdf"])

# --- Helper: Ensure event loop exists ---
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# --- Helper: Safe embedding with retry ---
@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(6))
def safe_add_docs(vectorstore, docs):
    try:
        vectorstore.add_documents(docs)
    except ResourceExhausted as e:
        st.warning(f"⚠️ Rate limit exceeded, retrying: {e}")
        raise
    except InvalidArgument as e:
        st.warning(f"⚠️ Chunk too big or bad input: {e}")
        raise
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
        raise

# --- Initialize embeddings only (no LLM yet) ---
def init_embeddings():
    ensure_event_loop()
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# --- Dynamic Chroma DB Path ---
def get_new_chroma_path(base_path="./chroma_db"):
    i = 1
    while os.path.exists(f"{base_path}{i}"):
        i += 1
    return f"{base_path}{i}"

if uploadedfile:
    # Step 1: Load PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploadedfile.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    st.success(f"✅ Loaded {len(docs)} pages")

    # Step 2: Split docs into chunks
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    st.info(f"✅ Split into {len(chunks)} chunks")

    # Step 3: Init embeddings
    embeddings = init_embeddings()

    # Step 4: Store embeddings in vector DB
    if ENVIRONMENT == "local":
        db_path = get_new_chroma_path()
        st.info(f"Using Chroma vector store (local) → {db_path}")
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            with st.spinner(f"🔄 Embedding batch {i//batch_size + 1}..."):
                try:
                    safe_add_docs(vectorstore, batch)
                except Exception as e:
                    st.error(f"Batch {i//batch_size + 1} failed in {db_path}: {e}")
                    # Create fallback DB
                    db_path = get_new_chroma_path()
                    st.warning(f"➡️ Switching to new DB path: {db_path}")
                    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
                    try:
                        safe_add_docs(vectorstore, batch)
                        st.success(f"✅ Batch stored in fallback DB {db_path}")
                    except Exception as e2:
                        st.error(f"❌ Even fallback DB failed for batch {i//batch_size + 1}: {e2}")
                time.sleep(2)

        vectorstore.persist()
        st.success(f"✅ Documents stored in Chroma DB at {db_path}")

    else:
        st.info("Using Supabase vector store (production)")
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        vectorstore = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents",
        )

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            with st.spinner(f"🔄 Uploading batch {i//batch_size + 1} to Supabase..."):
                try:
                    safe_add_docs(vectorstore, batch)
                except Exception as e:
                    st.error(f"❌ Supabase batch {i//batch_size + 1} failed: {e}")
                time.sleep(2)

        st.success("✅ Documents stored in Supabase")

    # ✅ Step 5: Init LLM *after* vectorstore is ready
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2
    )
    retriever = vectorstore.as_retriever()

    # Step 6: Build QA chain
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use only the following context to answer "
        "the question. "
        "If the context does not contain the answer, respond with: 'Sorry, question is not related to the document'. "
        "Limit your answer to no more than 10 words.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # Step 7: Ask questions
    query = st.text_input("🔎 Ask a question:")
    if query:
        with st.spinner("Processing..."):
            result = rag_chain.run(query)
            st.write("### 📌 Result:", result)
else:
    st.info("⬆️ Please upload a PDF to start.")
