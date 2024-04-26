# %%
import utils
import pandas as pd
import numpy as np
import hashlib

def get_author_name(book_id, data):
    authors_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    author_name = pd.melt(authors_df).query("value == @book_id").variable.tolist()[0]
    return author_name

# %%
# Load in datasets
book_contents_train = utils.load_book_contents(utils.book_authors_train)
book_contents_test = utils.load_book_contents(utils.book_authors_test)
# Tokenize the data
books_train_wtoks = utils.wtok_books(book_contents_train)
books_test_wtoks = utils.wtok_books(book_contents_test)
# Get author names
author_names_train = {k:get_author_name(k, utils.book_authors) for k in book_contents_train.keys()}
author_names_test = {k:get_author_name(k, utils.book_authors_test) for k in book_contents_test.keys()}

# %%
train_df = (
    pd.DataFrame([book_contents_train, books_train_wtoks, author_names_train])
    .T
    .reset_index()
    .assign(cond='train')
)
train_df.columns = ['book_id', 'contents', 'words', 'author_name', 'cond']
test_df = (
    pd.DataFrame([book_contents_test, books_test_wtoks, author_names_test])
    .T
    .reset_index()
    .assign(cond='test')
)
test_df.columns = ['book_id', 'contents', 'words', 'author_name', 'cond']
df_combined = pd.concat([train_df, test_df], axis=0)
# Get the author id as a number
myhash=lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest()[0:5], 16)
df_combined['author_id'] = (
    df_combined.author_name
    .apply(lambda x: myhash(x) % 100)
)
# Make sure it is between 0 and 2
idxs = sorted(df_combined.author_id.unique())
df_combined['author_id'] = (
    df_combined.author_id
    .apply(lambda x: idxs.index(x))
)

# %%
df_combined.to_pickle("../data/data.pkl")
# %%
