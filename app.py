import streamlit as st
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_from_files(uploaded_files):
    texts = []
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            texts.append((file.name, extract_text_from_pdf(file)))
        else:
            texts.append((file.name, str(file.read(), 'utf-8')))
    return texts

def rank_resumes(resume_texts, job_desc):
    documents = [job_desc] + [text for _, text in resume_texts]
    tfidf = TfidfVectorizer().fit_transform(documents)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    ranked = sorted(zip(resume_texts, scores), key=lambda x: x[1], reverse=True)
    return ranked

st.title("AI-powered Resume Screening System")

job_desc = st.text_area("Paste Job Description Here")

uploaded_files = st.file_uploader("Upload Resumes (PDF or TXT)", accept_multiple_files=True)

if st.button("Screen Resumes") and uploaded_files and job_desc:
    resume_texts = get_text_from_files(uploaded_files)
    ranked = rank_resumes(resume_texts, job_desc)
    
    st.subheader("Ranked Resumes")
    for (filename, _), score in ranked:
        st.write(f"**{filename}** — Score: {round(score * 100, 2)}%")
