import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# Configure Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Please set it in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Avoid NoneType errors
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split extracted text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Generate embeddings and store them in FAISS vector database."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")  # Save FAISS index locally
        st.success("Vector database created successfully!")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")

def get_conversational_chain():
    """Create an AI conversational chain for answering user questions."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say: "Answer is not available in the context".
    Don't provide incorrect information.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",  # Updated model
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """Handles user queries by retrieving relevant data and generating a response."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply:", response.get("output_text", "No response generated."))
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with your PDFs")

    user_question = st.text_input("Ask a question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and click on Submit & Process", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done. You can now ask questions.")
                else:
                    st.warning("No text extracted from PDFs. Please check the file content.")

if __name__ == "__main__":
    main()
