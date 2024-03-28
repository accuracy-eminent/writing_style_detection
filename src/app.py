import streamlit as st
import pandas as pd
import numpy as np
import utils
import model as md

def classify(text):
    print(text)
    clf = md.TabularModel()
    clf.load_weights("../data/tabular.pkl")
    preds = clf.predict(text)
    return preds
    #return len(text)

st.title("Detection of writing style")
st.write("Was the text written by Herman Melville, Jane Austen, or Charles Dickens?")

text_box = st.text_area('Text to classify', placeholder='Enter text here')

if st.button('Classify'):
    chars = classify(text_box)
    st.write(f"There are {chars} characters in the text")