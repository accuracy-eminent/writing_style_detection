import streamlit as st
import pandas as pd
import numpy as np
import utils
import model as md

def classify(text):
    # 0:cd 1:ja 2:hm 
    nn = True
    print(text)
    vocab_df = pd.read_csv("../data/vocab.csv")
    vocab_loaded = pd.Series(vocab_df.num.tolist(), index=vocab_df.word.tolist()).to_dict()
    if nn:
        clf = md.NNModel(vocab=vocab_loaded)
        clf.load_weights("../data/nn.pth")
    else:
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