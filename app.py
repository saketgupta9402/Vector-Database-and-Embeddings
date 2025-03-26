import streamlit as st
from PyPDF2 import PdfReader

import os

import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
else:
    raise ValueError("API key not found. Please set GOOGLE_API_KEY in your environment variables.")

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = "" 
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure text is not None before appending
                text += page_text  
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

# Create vector store
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load conversation model
def get_conversational_chain():
    prompt_template="""Answer the question as detailed as possible from the provided context, make sure to provide 
    all the details, if the answer is not in the provided context just say, "answer is not available in the context",
    dont't provide the wrong answer\n\n
    Context: \n {context}?\n
    Question: \n {question}\n
    
    Answer:
    """
    model= ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    
    return chain

# Handle user input
def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists("faiss_index"):
        st.error("No FAISS index found. Please upload a PDF first.")
        return

    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)

    chain=get_conversational_chain()

    response=chain( {"input_documents":docs, "question":user_question}, return_only_outputs=True)

    print(response)

    st.write("Reply: ",response.get("output_text","No response generated."))

# Main function
def main():
    st.set_page_config(page_title="Chat with your PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF file")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("""Upload Your PDF files and click on the 'Submit and Process' button""",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing"):
                raw_text= get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

if __name__=="__main__":
    main()