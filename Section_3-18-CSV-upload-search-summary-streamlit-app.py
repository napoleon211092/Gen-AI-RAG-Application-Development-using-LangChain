#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI


# In[22]:


# Setup model
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key


# In[23]:


#from langchain.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import textwrap
import streamlit as st


# In[24]:


def process_docx(docx_file):
    # Add your docx processing code here
    text=""
    # Docx2txtLoader loads the document
    loader = Docx2txtLoader(docx_file)
    # Load documentsand split into chunks
    text = loader.load_and_split()
    return text
def process_pdf(pdf_file):
    text = ""
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    for page in pages:
        text += page.page_content
    text = text.replace("\t"," ")
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 50
    )
    # create_document() create documents from a list of texts
    texts = text_splitter.create_documents([text])
    return texts


# In[25]:


def main():
    st.title("CV Summary Generator")
    uploaded_file = st.file_uploader("Select CV",type=["docx","pdf"])
    text = ""
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        st.write("File details:")
        st.write("File name: {uploaded_file.name}")
        st.write("File type: {file_extension}")
        if file_extension == "docx":
            text = process_docx(uploaded_file.name)
        elif file_extension == "pdf":
            text = process_pdf(uploaded_file.name)
        else:
            st.error("Unsupported file format. Please upload a .docx or .pdf file.")
            return
        llm = ChatOpenAI(temperature=0)
        prompt_template = """You have been given a Resume to analyze.
        Write a verbose detail of the following:
        {text}
        Details:"""
        prompt = PromptTemplate.from_template(prompt_template)
        refine_template = (
            "Your job is to produce a final outcome\n"
            "We have provided an existing detail: {existing_answer}\n"
            "We want a refined version of the existing detail based on initial details below\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary in the following manner:"
            "Name: \n"
            "Email: \n"
            "Key Skills: \n"
            "Last Company: \n"
            "Experience Summary: \n"
        )
        refine_prompt = PromptTemplate.from_template(refine_template)
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt = prompt,
            refine_prompt = refine_prompt,
            return_intermediate_steps = True,
            input_key = "input_documents",
            output_key = "output_text"
        )
        result = chain.invoke({"input_documents":text},return_only_outputs=True)
        st.write("Resume summary:")
        st.text_area("Text",result["output_text"], height=400)


# In[27]:


if __name__ == "__main__":
    main()
# run command
# cd C:\Users\ASUS\anaconda3\Gen-AI-RAG-Application-Development-using-LangChain
# streamlit run Section_3-18-CSV-upload-search-summary-streamlit-app.py --server.enableXsrfProtection false
