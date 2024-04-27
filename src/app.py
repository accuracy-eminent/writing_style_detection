import streamlit as st
import pandas as pd
import numpy as np
import utils
import model as md

# Load in author names
authors = pd.read_pickle("../data/data.pkl")
authors = (
    pd.read_pickle("../data/data.pkl")
    .filter(['author_name','author_id'])
    .drop_duplicates()
    .set_index('author_id')
    .author_name.to_dict()
)
# Load the model
vocab_df = pd.read_csv("../data/vocab.csv")
vocab_loaded = pd.Series(vocab_df.num.tolist(), index=vocab_df.word.tolist()).to_dict()
clf = md.NNModel(vocab=vocab_loaded)
clf.load_weights("../data/nn.pth")


st.title("Detection of writing style")
authors_string = ", ".join(list(authors.values())[:-1])
authors_string += ", or " + list(authors.values())[-1]
st.write(f"Was the text written by {authors_string}?")

text_box = st.text_area('Text to classify', placeholder='Enter text here')

if st.button('Classify'):
    pred_prob = clf.predict(text_box)
    pred_author = authors[np.argmax(pred_prob)]
    pred_prob = np.max(pred_prob)
    st.write(f"We are {100*pred_prob:.0f}% sure the text was written by {pred_author}")