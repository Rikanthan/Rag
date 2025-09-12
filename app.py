import os
import tempfile
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

load_dotenv()

# --- Config ---
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print(GOOGLE_API_KEY)

# --- Streamlit UI ---
st.title("📚 RAG App: Chroma (local) / Supabase (prod)")

uploadedfile = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploadedfile:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploadedfile.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    st.success(f"✅ Loaded {len(docs)} pages")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # --- Choose vector store based on environment ---
    if ENVIRONMENT == "local":
        st.info("Using Chroma vector store (local)")
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory="./chroma_db"  # local folder
        )
    else:
        st.info("Using Supabase vector store (production)")
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        vectorstore = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents",
        )
        # store documents in Supabase
        vectorstore.add_documents(chunks)
        st.success("✅ Documents stored in Supabase")

    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2
    )

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

    query = st.text_input("🔎 Ask a question:")
    if query:
        with st.spinner("Processing..."):
            result = rag_chain.run(query)
            st.write("### 📌 Result:", result)
else:
    st.info("⬆️ Please upload a PDF to start.")
