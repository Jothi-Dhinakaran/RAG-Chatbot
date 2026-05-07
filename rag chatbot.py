
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Page config ──
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 RAG Chatbot")
st.caption("Ask anything about your uploaded documents")

# ── Session state ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []


# ── Document loader ──
def load_document(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    tmp_path = f"tmp_{uploaded_file.name}"

    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    if ext == ".txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(tmp_path)
    else:
        st.error(f"Unsupported file: {uploaded_file.name}")
        return None

    return loader.load()


# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files and not st.session_state.ready:
        with st.spinner("Processing documents..."):
            all_docs = []
            for file in uploaded_files:
                docs = load_document(file)
                if docs:
                    all_docs.extend(docs)

            if all_docs:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                splitter = SemanticChunker(
                    embeddings,
                    breakpoint_threshold_type="percentile"
                )
                chunks = splitter.split_documents(all_docs)
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.ready = True
                st.session_state.uploaded_names = [f.name for f in uploaded_files]
                st.success(f"Ready! {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.ready:
        st.divider()
        st.caption("📂 Indexed files:")
        for name in st.session_state.uploaded_names:
            st.info(f"📄 {name}")

    st.divider()
    st.caption("Model: Gemini 2.5 Flash")
    st.caption("Embeddings: MiniLM-L6-v2")
    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset & Upload New"):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.ready = False
        st.session_state.uploaded_names = []
        st.rerun()


# ── LLM + Chains ──
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY,
)

rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite as a standalone question. Only give the rewritten question.\n"
    "Chat history: {chat_history}\nQuestion: {question}"
)
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_template(
    "Use the context to answer. If unsure, say you don't know.\n"
    "Context: {context}\nChat history: {chat_history}\nQuestion: {question}"
)
answer_chain = answer_prompt | llm | StrOutputParser()

eli5_prompt = ChatPromptTemplate.from_template(
    "Explain this answer like I'm 5 years old in 2-3 simple sentences:\n{answer}"
)
eli5_chain = eli5_prompt | llm | StrOutputParser()


# ── Chat display ──
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "eli5" in msg:
            st.info(f"🧒 ELI5: {msg['eli5']}")


# ── Chat input ──
if not st.session_state.ready:
    st.warning("👈 Upload one or more documents from the sidebar to start chatting.")
else:
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-6:]]
        )

        rewritten = rewrite_chain.invoke({
            "chat_history": history_text,
            "question": user_input
        })

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        context_docs = retriever.invoke(rewritten)
        context = "\n\n".join([d.page_content for d in context_docs])

        answer = answer_chain.invoke({
            "context": context,
            "chat_history": history_text,
            "question": user_input
        })

        eli5 = eli5_chain.invoke({"answer": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.info(f"🧒 ELI5: {eli5}")

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "eli5": eli5
        })