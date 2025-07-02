import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a scientific research assistant.
Given the following context from scientific papers, answer the user's question.
Be concise, technical, and include any relevant research suggestions.

Context:
{context}

Question:
{question}

Answer:
"""
)

# Load and split PDF
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Create vector store
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# Build QA chain
def build_qa_chain(vectorstore):
    llm = ChatGroq(model_name="qwen-qwq-32b", temperature=0, api_key=api_key)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Q&A Assistant", layout="centered")
    st.title("📄 Scientific Paper Summarization Assistant ")

    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    question = st.text_input("Ask a question about the document")

    if uploaded_pdf and question:
        with st.spinner("Processing..."):
            # Save uploaded PDF to temp file
            os.makedirs("temp", exist_ok=True)
            temp_path = os.path.join("temp", uploaded_pdf.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.read())

            # Rebuild vector store from the uploaded file
            docs = load_and_split_pdf(temp_path)
            vectorstore = create_vector_store(docs)
            qa_chain = build_qa_chain(vectorstore)

            # Get answer
            answer = qa_chain.run(question)

            # Show result
            st.success("Answer generated:")
            st.markdown(f"**💡 Response:** {answer}")

if __name__ == "__main__":
    main()
