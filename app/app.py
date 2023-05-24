# python3 -m streamlit run app.py

import streamlit as st
from PIL import Image
import numpy as np
import yolov5
from pathlib import Path
import shutil

dirpath = Path.cwd() / 'results'
model_path = Path.cwd() / 'models'
#print(dirpath)
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)


st.title("VQArt")


st.markdown("""Hello, please take a picture of the painting and ask a question about it. \\ 
               I can answer questions about the style, artist and genre of the painting, \\
               and then questions about these topics. \\
               """)

    

# load model

# Take a picture
imgbuffer = st.camera_input('')

# Prompt for a question
st.markdown("What is your question (e.g. Who's the author of the painting?)")

if imgbuffer:
    img = Image.open(imgbuffer)
    imgnp = np.array(img)
    #st.write(imgnp.shape)

    # perform inference
    results = model(img, size=640)
    st.markdown("The artist of this painting is Leonardo Da Vinci")


    # Information extraction