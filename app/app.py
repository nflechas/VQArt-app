# python3 -m streamlit run app.py

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
import sys
sys.path.insert(1, "src/models")
from extractive_qa import QA
from search_engine import IR
# from src.models.extractive_qa import QA
# from src.models.search_engine import IR

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

# Defining a session variable
if 'extractive_qa' not in st.session_state:
	st.session_state.extractive_qa = False

dirpath = Path.cwd() / 'results'
model_path = Path.cwd() / 'models'
#print(dirpath)
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

qa_module = load_qa_module()
search_engine = load_search_engine()

st.title("VQArt")

st.markdown("""Hello, please take a picture of the painting and ask a question about it. \
               I can answer questions about the style, artist and genre of the painting, \
               and then questions about these topics. \
               """)

# Take a picture
imgbuffer = st.camera_input('')

# Prompt for a question
question = st.text_input(label="What is your question (e.g. Who's the author of the painting?")

if question:
    print(f'Received question: {question}')

    if st.session_state.extractive_qa:
        # Doing Extractive QA
        articles, scores = search_engine.retrieve_documents(question, 5)
        print(f'Found {len(articles)} search results')
        
        if len(articles) == 0:
            st.write("Sorry, I don't know the answer to that question :(")
        else:
            best_result = articles[0]
            answer = qa_module.answer_question(question, best_result)
            st.write(answer)
    else:
        # Doing VQA
        st.write('answer to vqa')

        # Switching to extractive QA
        st.session_state.extractive_qa = True

if imgbuffer:
    img = Image.open(imgbuffer)
    imgnp = np.array(img)
    #st.write(imgnp.shape)

    # perform inference
    results = model(img, size=640)
    st.markdown("The artist of this painting is Leonardo Da Vinci")

    # Information extraction