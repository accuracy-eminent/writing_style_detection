import streamlit as st
import pandas as pd
import numpy as np
import utils
import model as md

#authors = ['Charles Dickens', 'Jane Austen', 'Herman Melville']
authors = None
authors = pd.read_pickle("../data/data.pkl")
authors = (
    pd.read_pickle("../data/data.pkl")
    .filter(['author_name','author_id'])
    .drop_duplicates()
    .set_index('author_id')
    .author_name.to_dict()
)

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
    pred_prob = classify(text_box)
    pred_author = authors[np.argmax(pred_prob)]
    pred_prob = np.max(pred_prob)
    st.write(f"We are {100*pred_prob:.0f}% sure the text was written by {pred_author}")