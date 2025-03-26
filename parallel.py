import streamlit as st
from pdfminer.high_level import extract_text
import os
import concurrent.futures
import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

# Load API Key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
else:
    raise ValueError("API key not found. Please set GOOGLE_API_KEY in your environment variables.")

# 🏃 **Ultra-Fast PDF Text Extraction**
def get_pdf_text(pdf_docs):
    text = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda pdf: extract_text(pdf), pdf_docs)
        text.extend(filter(None, results))
    return " ".join(text)

# 🔄 **Optimized Text Splitting**
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)  # Smaller chunks for speed
    return text_splitter.split_text(text)

# ⚡ **Batch FAISS Vector Storage for Speed**
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store  # Keep in memory instead of saving to disk

# 📜 **Load LLM Chain**
def get_conversational_chain():
    prompt_template = """Answer the question based on the given context. If the answer is not available, reply: "Answer is not available in the context."
    
    Context: {context} 
    Question: {question} 
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# 📥 **Handle User Questions**
def user_input(user_question, vector_store):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response.get("output_text", "No response generated.")

# 🚀 **Optimized Main Function**
def main():
    st.set_page_config(page_title="Chat with your PDF")
    st.header("Chat with PDF using Gemini")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Your PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)  # Use PDFMiner (Super fast)
                text_chunks = get_text_chunks(raw_text)  # Reduce chunk size for speed
                st.session_state.vector_store = get_vector_store(text_chunks)  # Keep FAISS in memory
                st.success("Processing complete!")

    # Continuous chat interface
    user_question = st.chat_input("Ask a Question from the PDF file (Type 'exit' to quit)")
    if user_question:
        if user_question.lower() == "exit":
            st.write("Exiting chat. Thank you!")
            return
        
        if st.session_state.vector_store is None:
            st.error("No FAISS index found. Please upload and process a PDF first.")
            return
        
        reply = user_input(user_question, st.session_state.vector_store)
        st.write(f"**Bot:** {reply}")

if __name__ == "__main__":
    main()
