# %%
import utils
import pandas as pd
import numpy as np

# %%
# Load in datasets
book_contents_train = utils.load_book_contents(utils.book_authors_train)
book_contents_test = utils.load_book_contents(utils.book_authors_test)
# Tokenize the data
books_train_wtoks = utils.wtok_books(book_contents_train)
books_test_wtoks = utils.wtok_books(book_contents_test)

# %%
train_df = (
    pd.DataFrame([book_contents_train, books_train_wtoks])
    .T
    .reset_index()
    .assign(cond='train')
)
train_df.columns = ['book_id', 'contents', 'words', 'cond']
test_df = (
    pd.DataFrame([book_contents_test, books_test_wtoks])
    .T
    .reset_index()
    .assign(cond='test')
)
train_df.columns = ['book_id', 'contents', 'words', 'cond']



# %%
