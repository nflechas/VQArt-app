# python3 -m streamlit run app.py

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
import sys
sys.path.insert(1, "src/models")
from extractive_qa import QA
from visual_qa import VisualQA
from search_engine import IR
# from src.models.extractive_qa import QA
# from src.models.search_engine import IR


@st.cache_resource
def load_visual_qa_module():
    """
    Loads the Visual QA module
    """
    qa_module = VisualQA()
    return qa_module

@st.cache_resource
def load_qa_module():
    """
    Loads the extractive QA module
    """
    qa_module = QA()
    return qa_module

@st.cache_resource
def load_search_engine():
    """
    Loads the extractive QA module
    """
    search_engine = IR()
    return search_engine

def get_metadata_from_question(question):
    if 'artist' in question:
        return 'artist'
    elif 'style' in question:
        return 'style'
    elif 'genre' in question:
        return 'genre'

# Defining session variables
if 'extractive_qa' not in st.session_state:
    st.session_state.extractive_qa = False

if 'vqa_prediction' not in st.session_state:
    st.session_state.vqa_prediction = None

dirpath = Path.cwd() / 'results'
model_path = Path.cwd() / 'models'
#print(dirpath)
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

vqa_module = load_visual_qa_module()
qa_module = load_qa_module()
search_engine = load_search_engine()

st.title("VQArt")

st.markdown("""Hello, please take a picture of the painting and ask a question about it. \
               I can answer questions about the style, artist and genre of the painting, \
               and then questions about these topics. \
               """)

# Take a picture
imgbuffer = st.camera_input('')

# Upload a file
uploaded_file = st.file_uploader('Upload a photo of a painting')

# Prompt for a question
question = st.text_input(label="What is your question (e.g. Who's the artist of this painting?)")

if question:
    print(f'Received question: {question}')

    if st.session_state.extractive_qa:
        # Doing Extractive QA
        full_question = f'[{st.session_state.vqa_prediction}] {question}'

        articles, scores = search_engine.retrieve_documents(full_question, 5)
        print(f'Found {len(articles)} search results')
        
        if len(articles) == 0:
            st.markdown("Sorry, I don't know the answer to that question :(")
        else:
            best_result = articles[0]
            answer = qa_module.answer_question(full_question, best_result)
            st.markdown(f'Answer: {answer}')
    else:
        # Doing VQA

        if imgbuffer:
            # Camera
            img = Image.open(imgbuffer)
        elif uploaded_file:
            # Uploaded file
            img = Image.open(uploaded_file)

        result = vqa_module.answer_question(question, img)
        meta_data = get_metadata_from_question(question)
        st.markdown(f"Answer: The {meta_data} of this painting is {result}")

        # Switching to extractive QA
        st.session_state.extractive_qa = True

        # Saving the predicted VQA answer
        st.session_state.vqa_prediction = result