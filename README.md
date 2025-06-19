# CHAT WITH PDF
This Streamlit web app allows users to upload multiple PDF files and ask questions about the content. The app extracts text from PDFs, processes it using Google Generative AI embeddings, and responds with accurate answers using the Gemini LLM.

## Features
- Upload multiple PDF files.
- Extract and chunk text for processing
- Store data in a FAISS vector database
- Store data in a FAISS vector database
- Gemini 1.5 Pro used for accurate responses
- Built with LangChain, Streamlit, and Google Generative AI

## Tech Stack
- **Frontend**: Streamlit
- **LLM**: Gemini 1.5 Pro (via Google Generative AI)
- **Vector Store**:FAISS
- **PDF Parsing**:PyPDF2
- **Embeddings**:GoogleGenerativeAIEmbeddings
  
