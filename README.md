# RAG based Scientific Paper Summarization and Suggestion 

This project is a Streamlit-based web application that allows users to upload scientific PDFs and ask questions based on the content. It leverages LangChain, HuggingFace embeddings, FAISS vector store, and the Groq LLM for retrieval-augmented question answering.

## Features

- Upload and parse PDF documents
- Generate vector embeddings using HuggingFace
- Store and retrieve context using FAISS
- Ask natural language questions about the uploaded content
- Get concise, technical answers using the Groq language model

## Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Embeddings](https://huggingface.co/)
- [Groq LLM](https://groq.com/)
- [Python-dotenv](https://pypi.org/project/python-dotenv/)

## Usage

1. Launch the app in your browser.
2. Upload a scientific PDF document.
3. Ask questions related to the content.
4. Receive answers backed by extracted context from the PDF.

## Notes

- The prompt is designed for scientific research questions.
- Only one PDF is handled at a time.
- Make sure the content of the PDF is extractable (not scanned images).

