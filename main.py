
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
from google_web_scrapper_bs4 import *
from tavily_data_generator import *
from selenium_webscrapper import *
from langchain.prompts import PromptTemplate
import logging
logging.disable(logging.WARNING)
#C:\Users\athar\PycharmProjects\WebScrapping\.venv\countries_info.pdf
import PyPDF2
import gc
import numpy as np
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import time
import google.generativeai as genai
os.environ["GEMINI_API_KEY"] = "AIzaSyDshCBROyoi1dByysldqps8mQ8_wN_9YuA "  # Replace with your API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
chunks = []
nlp = spacy.load("en_core_web_sm")
def read_and_extract_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please provide a valid PDF path.")
        return None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None



def build_rag_model(text):

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    chunks.extend(splitter.split_text(text))

    gc.collect()


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    batch_size = 100
    vector_store = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_texts(batch, embeddings)
        else:
            temp_store = FAISS.from_texts(batch, embeddings)
            vector_store.merge_from(temp_store)
            del temp_store  # Free memory
        gc.collect()
    return vector_store, embeddings


def add_new_text_to_model(new_text, vector_store, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    new_chunks = splitter.split_text(new_text)
    #chunks.extend(new_chunks)

    batch_size = 100
    temp_store = None
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        if temp_store is None:
            temp_store = FAISS.from_texts(batch, embeddings)
        else:
            batch_store = FAISS.from_texts(batch, embeddings)
            temp_store.merge_from(batch_store)
            del batch_store
        gc.collect()

    vector_store.merge_from(temp_store)
    del temp_store
    gc.collect()
    return vector_store
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="""You are a wizard who  like to answers questions based on the content of a PDF document in a wicked wizard manner with spells. Use the following context to answer the question and answer in full full sentences. No one word answeres. If the context does not contain enough information to answer confidently, say "I cannot answer this question based on the provided context."

Context:
{context}

Question:
{query}

Answer:
"""
)




def setup_rag_pipeline(vector_store):

    model = genai.GenerativeModel('gemini-1.5-flash')
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return model, retriever

def extract_key_terms(query):
    doc = nlp(query.lower())

    key_terms = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] or token.ent_type_:
            key_terms.append(token.text)

    return list(dict.fromkeys(key_terms))
def cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def get_significant_terms(chunks, top_n=50):

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(chunks)
    feature_names = vectorizer.get_feature_names_out()

    avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)

    top_indices = avg_scores.argsort()[-top_n:][::-1]
    significant_terms = [feature_names[idx] for idx in top_indices]
    return set(significant_terms)

def can_confidently_answer(query, chunks, embeddings, vector_store, threshold=0.7):

    key_terms = extract_key_terms(query)
    if not key_terms:
        print(0)
        return False

    pdf_terms = get_significant_terms(chunks)


    full_text = " ".join(chunks).lower()
    term_presence = sum(1 for term in key_terms if term in full_text)
    term_presence_ratio = term_presence / len(key_terms)


    context_overlap = sum(1 for term in key_terms if term in pdf_terms)
    context_overlap_ratio = context_overlap / len(key_terms) if key_terms else 0.0

    query_embedding = embeddings.embed_query(query)
    chunk_embeddings = vector_store.index.reconstruct_n(0, vector_store.index.ntotal)
    similarities = [cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
    semantic_similarity_score = max(similarities) if similarities else 0.0

    confidence_score = (0.25 * term_presence_ratio) + (0.25 * context_overlap_ratio) + (0.5 * semantic_similarity_score)
    print(f"Term Presence Ratio: {term_presence_ratio:.2f}")
    print(f"Context Overlap Ratio: {context_overlap_ratio:.2f}")
    print(f"Semantic Similarity Score: {semantic_similarity_score:.2f}")
    print(f"Combined Confidence Score: {confidence_score:.2f}")

    print(confidence_score)
    can_answer = confidence_score >= threshold
    return can_answer
def set_scrapper(scrapper):
    print(f"Setting scrapper to {scrapper}")
    if scrapper == "bs4":
        return bs4_web_scrapper
    elif scrapper == "tavily":
        return tavily_answer_generator
    else:
        return selenium_web_scrapper
def run_chatbot():

    file_path = input("Please enter the path to your PDF file (e.g., document.pdf): ")
    pdf_text = read_and_extract_pdf(file_path)
    if pdf_text is None:
        print("Exiting due to error in PDF processing.")
        return

    print(f" {file_path} extracted successfully.")

    vector_store, embeddings = build_rag_model(pdf_text)
    print("RAG model prepared successfully.")


    qa_chain = setup_rag_pipeline(vector_store)
    scrapper = input("Please select your webscrapper (1. bs4  2.tavily 3.selenium):")
    web_scrapper = set_scrapper(scrapper)
    print(f"Web Integrated using {scrapper} PDF-Chatbot is ready! Ask questions about the PDF content (type 'exit' to stop).")
    #print("1")
    while True:
        #print("2")
        time.sleep(2)
        query = input("Your question: ")
        print(query)
        if query.lower() == "exit":
            print("Thank You For Using the Web Integrated PDF Chatbot")
            break
        if query.lower() == "change":
            scrapper = input("Please select your webscrapper (1. bs4  2.tavily 3.selenium):")
            web_scrapper = set_scrapper(scrapper)
            continue
        if not query.strip():
            print("Please enter a valid question.")
            continue
        model, retriever = setup_rag_pipeline(vector_store)
        docs = retriever.invoke(query)
        if not docs:
            print("No relevant context found in the PDF.")
            continue
        context = " ".join([doc.page_content for doc in docs])


        formatted_prompt = prompt_template.format(context=context, query=query)
        print("Formatted Prompt:\n", formatted_prompt)


        response = model.generate_content(formatted_prompt)
        answer = response.text

        print(answer)

        if can_confidently_answer(query, chunks, embeddings, vector_store)  == False :
            print("But I am not confident about this Answer, based on the Input PDF. I am going to use the web, Just a second.")
            new_text = web_scrapper(query)
            #print("4")
            if new_text == 0:
                print("Couldn't find any better answer")
            else:
                #print("5")
                vector_store = add_new_text_to_model(new_text, vector_store, embeddings)
                model, retriever = setup_rag_pipeline(vector_store)
                docs = retriever.invoke(query)
                if not docs:
                    print("No relevant context found in the PDF.")
                    continue
                context = " ".join([doc.page_content for doc in docs])

                # Manually format the prompt using the prompt_template
                formatted_prompt = prompt_template.format(context=context, query=query)
                print("Formatted Prompt:\n", formatted_prompt)  # Debug: Show the prompt

                # Invoke the LLM directly
                response = model.generate_content(formatted_prompt)
                answer = response.text

                print(f"Answer: {answer}\n")






# Run the chatbot
if __name__ == "__main__":
    run_chatbot()