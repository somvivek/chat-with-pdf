import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from pypdf import PdfReader

# ---------------- PDF RAG Functions ----------------

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def split_pdf_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_vectorstore(chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

def create_conversational_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the context below to answer the question.\n\n"
        "Don't make up answers, only use the provided context.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_type="stuff",
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        },
        return_source_documents=True  # ✅ important to ensure "source_documents" is returned
    )
    return chain

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="💬 Chat with PDF (Memory)", page_icon="🧠")
st.title("🧠 Chat with your PDF using LangChain Memory")

openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
uploaded_pdf = st.file_uploader("📎 Upload a PDF file", type=["pdf"])

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state:
    st.session_state.chain = None

# Step 1: Process PDF
if openai_api_key and uploaded_pdf and st.session_state.chain is None:
    with st.spinner("Processing PDF..."):
        try:
            text = extract_text_from_pdf(uploaded_pdf)
            chunks = split_pdf_text(text)
            vectorstore = create_vectorstore(chunks, openai_api_key)
            st.session_state.chain = create_conversational_chain(vectorstore, openai_api_key)
            st.success("✅ PDF processed. You can now chat.")
        except Exception as e:
            st.error(f"Error while processing PDF: {e}")

# Step 2: Chat Interface
if st.session_state.chain:
    user_input = st.chat_input("Ask a question...")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.chain.invoke({"question": user_input})
                st.write("DEBUG result:", result)
                answer = result.get("answer", "No answer returned.")
                sources = result.get("source_documents", [])

                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("ai", answer))
                if sources:
                    st.session_state.chat_history.append(("meta", sources))

            except Exception as e:
                st.error(f"Error generating answer: {e}")

# Step 3: Render Chat History
for role, message in st.session_state.chat_history:
    with st.chat_message(role if role != "meta" else "ai"):
        if role == "meta":
            with st.expander("📄 Sources"):
                for i, doc in enumerate(message, 1):
                    st.markdown(f"**Source {i}:** {doc.page_content[:300]}...")
        else:
            st.markdown(message)

# Show initial instruction
if not openai_api_key or not uploaded_pdf:
    st.info("Please enter your API key and upload a PDF to begin chatting.")
