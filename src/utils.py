import pandas as pd
import numpy as np
import gutenbergpy.textget
import re
import string
from nltk import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from plotnine import *
from collections import Counter

# Create training and testing datasets
book_authors = {
    'Charles Dickens': [46, 98, 1400, 730, 766, 1023],
    'Herman Melville': [2701, 11231, 15859, 21816, 34970, 10712],
    'Jane Austen': [1342, 158, 161, 105, 121, 141]
}
book_authors_train = book_authors
book_authors_test = {
    'Charles Dickens': [786, 580, 883],
    'Herman Melville': [4045, 8118, 2694, 13720, 53861],
    'Jane Austen': [946, 1212]
}
book_authors_all = {}
for key in book_authors_train.keys():
    book_authors_all[key] = book_authors_train[key] + book_authors_test[key]

def get_book_authors_all():
    book_authors = {}
    for key in book_authors_train.keys():
        book_authors[key] = book_authors_train[key] + book_authors_test[key]
    return book_authors_all

def load_book_contents(book_authors):
    # Load in the text of the books from Project Gutenberg
    book_contents = {}
    for book_id in [book_id for id_list in book_authors.values() for book_id in id_list]:
        # Load in the book
        raw_book = gutenbergpy.textget.get_text_by_id(book_id)
        clean_book = gutenbergpy.textget.strip_headers(raw_book)
        # Convert to string
        book = clean_book.decode('UTF-8')
        # Save the book contents
        book_contents[book_id] = book
    return book_contents

def wtok_books(book_contents):
    # Tokenize each book into words
    books_wtoks = {}
    for book_id, book in book_contents.items():
        # Using RegexpTokenizer with \w+ keeps only alphabetic words
        # Using word_tokenize() keeps both alphabetic words and also counts puncutation as words
        #tokenizer = RegexpTokenizer(r'\w+')
        #books_wtoks[book_id] = tokenizer.tokenize(book)
        books_wtoks[book_id] = word_tokenize(book)
    return books_wtoks

def stok_books(book_contents):
    # Tokenize each book into sentences
    books_stoks = {}
    for book_id, book in book_contents.items():
        # Using RegexpTokenizer with \w+ keeps only alphabetic words
        # Using word_tokenize() keeps both alphabetic words and also counts puncutation as words
        #tokenizer = RegexpTokenizer(r'\w+')
        #books_wtoks[book_id] = tokenizer.tokenize(book)
        books_stoks[book_id] = sent_tokenize(book)
    return books_stoks

def calc_book_stats(book_contents, books_wtoks, books_stoks):
    # Calculate various stylometric metrics (characters per sentence, words per sentence, characters per word)
    book_stats = {}
    for book_id in book_contents.keys():
        book_stats[book_id] = {}
        book_stok = books_stoks[book_id]
        book_wtok = books_wtoks[book_id]
        (book_stats[book_id])['sent_chars'] = [len(sent) for sent in book_stok]
        (book_stats[book_id])['sent_words'] = [len(word_tokenize(sent)) for sent in book_stok]
        (book_stats[book_id])['word_chars'] = [len(word) for word in book_wtok]
    return book_stats

def stats_to_df(book_stats, book_authors):
    # Convert metrics to data frames
    stat_dfs = {}
    stat_names = book_stats[list(book_stats.keys())[0]]
    for stat_name in stat_names:
        stat_data = {book_id:book_stat[stat_name] for book_id, book_stat in book_stats.items()}
        stat_data_df = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in stat_data.items()}, orient='columns')
        stat_data_df_long = pd.melt(stat_data_df).dropna(how='any')
        stat_data_df_long.columns = ['book_id', stat_name]
        book_authors_df = pd.melt(pd.DataFrame.from_dict({k:pd.Series(v) for k, v in book_authors.items()}))
        book_authors_df.columns = ['author_name', 'book_id']
        stat_df_final = stat_data_df_long.merge(book_authors_df, on='book_id', how='left')
        stat_df_final = stat_df_final.sort_values('author_name')
        stat_dfs[stat_name] = stat_df_final
    return stat_dfs

def get_book_contents():
    # Load in the text of the books from Project Gutenberg
    book_contents = {}
    for book_id in [book_id for id_list in book_authors.values() for book_id in id_list]:
        # Load in the book
        raw_book = gutenbergpy.textget.get_text_by_id(book_id)
        clean_book = gutenbergpy.textget.strip_headers(raw_book)
        # Convert to string
        book = clean_book.decode('UTF-8')
        # Save the book contents
        book_contents[book_id] = book
    return book_contents

def get_ngrams(books_wtoks, book_authors, gram_length=3):
    book_grams = {}
    for author, books in book_authors.items():
        for book_id in books:
            top_grams = pd.Series(sorted(ngrams(books_wtoks[book_id], gram_length))).value_counts()
            book_grams[book_id] = top_grams
    return book_grams    


def combine_ngrams_by_author(book_authors, book_grams):
    author_ngs = {}
    for author, books in book_authors.items():
        ng_df = pd.DataFrame(book_grams[books[0]]).reset_index()
        ng_df.columns = ['ng', f'count_{books[0]}']
        for book_id in books[1:]:
            new_ng_df = pd.DataFrame(book_grams[book_id]).reset_index()
            new_ng_df.columns = ['ng', f'count_{book_id}']
            ng_df = ng_df.merge(new_ng_df, on='ng',how='outer')
            author_ngs[author] = ng_df
    return author_ngs

# Get ngrams that occur in all an author's work and have the highest odds ratio compared to the others
def get_likely_ngrams(author_ngs_combined, min_or=2.3):
    # Which ngrams are more probable than others by author?
    grams = (
        author_ngs_combined
        .query('all_cd or all_hm or all_ja')
        .filter(['ng','mean_cd','mean_ja','mean_hm'])
        .assign(mean_prob=lambda x: (x.mean_cd + x.mean_ja + x.mean_hm)/3)
    )
    # Calculate odds ratio of the odds of an ngram in one author's book compared to the mean
    likely_ngs = {}
    odds=lambda x: x / (1 - x)
    for author in ['cd','ja','hm']:
        grams[f'or_{author}'] = odds(grams[f'mean_{author}']) / odds(grams['mean_prob'])
        ngs = grams.sort_values(f'or_{author}', ascending=False).head(200).sort_values(f'mean_{author}').head(30)
        ngs = grams.query(f'or_{author} > {min_or}').sort_values(f'mean_{author}', ascending=False).head(30)
        likely_ngs[f'{author}'] = pd.Series(ngs.ng.tolist())
    return pd.DataFrame(likely_ngs)

# Find ngrams in each book and return the probability
def get_ngram_probabilities(gram_length, book_authors, books_wtoks):
    # Find ngrams in each book, and set their value to the probability of occurence
    book_grams = {}
    for author, books in book_authors.items():
        for book_id in books:
            sorted_ngrams = sorted(ngrams(books_wtoks[book_id], gram_length))
            top_grams = pd.Series(sorted_ngrams).value_counts() / len(sorted_ngrams)
            book_grams[book_id] = top_grams
    return book_grams

# Combine ngrams by author into a single data frame of probabilities of ngram occurence by book
def ngrams_by_author(book_grams, book_authors):
    author_ngs = {}
    # Combine ngrams by author into a single data frame, with each column being the probability of occurence in a each book
    for author, books in book_authors.items():
        #print(books[0])
        #print(book_grams)
        ng_df = pd.DataFrame(book_grams[books[0]]).reset_index()
        ng_df.columns = ['ng', f'count_{books[0]}']
        for book_id in books[1:]:
            new_ng_df = pd.DataFrame(book_grams[book_id]).reset_index()
            new_ng_df.columns = ['ng', f'count_{book_id}']
            ng_df = ng_df.merge(new_ng_df, on='ng',how='outer')
        author_ngs[author] = ng_df
    return author_ngs

# Create a combine data frame of ngrams for each author by book
def combine_ngrams(author_ngs):
    # Create a combined data frames of ngrams for each author,
    # with a column for mean number of times they appear in the author's work per book,
    # and a column saying whether this ngram shows up in all of the Author's books
    get_mean_words = lambda x: x.drop('ng', axis=1).median(axis=1)
    in_all_books = lambda x: x.drop('ng', axis=1).sum(axis=1, skipna=False) > 0
    ja_ng = author_ngs['Jane Austen'].assign(mean_ja=get_mean_words).assign(all_ja=in_all_books)
    hm_ng = author_ngs['Herman Melville'].assign(mean_hm=get_mean_words).assign(all_hm=in_all_books)
    cd_ng = author_ngs['Charles Dickens'].assign(mean_cd=get_mean_words).assign(all_cd=in_all_books)
    author_ngs_combined = (
        ja_ng
        .merge(cd_ng, on='ng', how='outer')
        .merge(hm_ng, on='ng', how='outer')
    )
    return author_ngs_combined

# Calculate ngrams and probabiliities for each author and book into a single data frame
def get_combined_ngram_df(gram_length, book_authors, books_wtoks):
    # Find ngrams in each book, with probabilities of occuranc
    book_grams = get_ngram_probabilities(gram_length, book_authors, books_wtoks)
    #print(book_grams)
    
    # Get ngrams grouped by author
    author_ngs = ngrams_by_author(book_grams, book_authors)

    # Combine the ngrams grouped by author into a single data frame
    author_ngs_combined = combine_ngrams(author_ngs)
    return author_ngs_combined

# Find ngrams unique to an author that occur in all of their books
def get_uniq_ngrams_all(author_ngs_combined):
    # Find ngrams that only appear in one author's work
    uniq_ngs = (
        author_ngs_combined
        .filter(['ng','mean_cd','mean_ja','mean_hm','all_cd','all_ja','all_hm'])
        # Find ngrams unique to one author
        .assign(books_with_ng=lambda x: (x.mean_cd > 0).astype(int) + (x.mean_ja > 0).astype(int) + (x.mean_hm > 0).astype(int))
        .query('books_with_ng == 1')
    )
    #print(uniq_ngs)
    # Get unique ngrams for each author that are present in all the author's books (f'all_{author} == True')
    uniq_all_ng = {}
    for author in ['cd','ja','hm']:
        ngs = uniq_ngs.query(f'mean_{author}> 0').query(f'all_{author} == True').sort_values(f'mean_{author}', ascending=False)
        # Convert to a list and back to a series to get the proper
        uniq_all_ng[author] = pd.Series(ngs.ng.tolist())
    return pd.DataFrame(uniq_all_ng)

def get_samples(books_wtoks, num_samples=100, sample_lengths=[500,1500], random_seed=42):
    book_samples = {}
    for book_id, words in books_wtoks.items():
        np.random.seed(random_seed)
        book_len = len(words)
        for i in range(0, num_samples):
            if sample_lengths[0] == sample_lengths[1]:
                sample_length = sample_lengths[0]
            else:
                sample_length = np.random.randint(low=sample_lengths[0], high=sample_lengths[1])
            start_idx = int(np.random.random(1)[0] * (book_len - sample_length))
            sample = words[start_idx:start_idx+sample_length]
            book_samples[f'{book_id}_{i}'] = sample
    return book_samples


# Use ngram frequency as features
# cd_1grams is the frequency of 1-grams associated with Charles Dickens, for example
def get_data_df(book_samples, book_authors):
    ref_grams = {}
    ref_grams[1] = {
        'cd': [('t',), ('don',), ('boy',), ('until',), ('stopped',), ('hair',), ('d',), ('streets',), ('shook',), ('shaking',)],
        'ja':[('her',), ('she',), ('She',), ('Mrs.',), ('herself',), ('sister',), ('father',), ('Lady',), ('wish',), ('Sir',)],
        'hm':[('sea',), ('strange',), ('THE',), ('Nor',), ('board',), ('ye',), ('ere',), ('peculiar',), ('concerning',), ('original',)]
    }
    ref_grams[2] = {
        'cd':[('’', 't'), ('don', '’'), (',', 'Mr.'), ('said', 'the'), ('his', 'head'), ('the', 'fire'), (',', 'looking'), ('I', 'said'), ('s', 'a'), ('“', 'Now')],
        'ja':[('.', 'She'), (',', 'she'), ('of', 'her'), ('she', 'had'), ('could', 'not'), ('to', 'her'), ('she', 'was'), ('that', 'she'), ('do', 'not'), ('she', 'could')],
        'hm':[(',', 'then'), (',', 'yet'), (';', 'in'), ('.', 'Nor'), ('so', 'that'), ('when', ','), ('.', 'Some'), ('though', ','), (';', 'while'), ('.', 'Upon')]
    }
    ref_grams[3] = {
        'cd': [('don', '’', 't'), ('!', '”', 'said'), ('?', '”', 'said'), ('’', 's', 'a'), ('.', '“', 'Now'), ('.', 'I', 'had'), ('as', 'if', 'he'), ('“', 'Now', ','), ('he', 'said', ','), ('.', '“', 'Yes')],
        'ja': [(',', 'however', ','), ('I', 'am', 'sure'), ('I', 'do', 'not'), (',', 'and', 'she'), ('.', 'She', 'was'), ('she', 'could', 'not'), ('.', 'She', 'had'), (',', 'she', 'was'), (';', 'and', 'she'), ('“', 'Oh', '!')],
        'hm': [(',', 'then', ','), (',', 'who', ','), ('.', 'But', 'the'), ('“', 'I', 'would'), (',', 'like', 'the'), ('that', ',', 'in'), (',', 'that', 'in'), ('answer', '.', '“'), ('out', 'of', 'sight'), (',', 'in', 'some')]
    }
    data_dict = {}
    for sample_id, words in book_samples.items():
        sample_row = {}
        get_ngrams = lambda words, gram_length: pd.Series(sorted(ngrams(words, gram_length))).value_counts()
        top_grams = {}
        # Calculate 1 to 3-grarms
        for gram_length in range(1, 4):
            top_grams[gram_length] = get_ngrams(words, gram_length)
        # Find the number of reference ngrams by author in each sample
        for author in ref_grams[1].keys():
            for gram_length in range(1, 4):
                top_grams_count = top_grams[gram_length]
                # Uese only the first 5 ngrams
                author_ref_grams = ref_grams[gram_length][author]#[0:5]
                author_grams_count = top_grams_count.reindex(author_ref_grams)
                # Normalize it by the length of the text
                sample_row[f'{author}_{gram_length}grams'] = author_grams_count.sum() / len(words)
        data_dict[sample_id] = sample_row
    # Create the initial data frame
    data_df = pd.DataFrame(data_dict).T
    data_df = (
        data_df
        .reset_index()
        .rename(columns={'index':'sample_id'})
    )
    # Clean data, attack to book authors 
    data_df = (
        data_df
        .assign(book_id=lambda x: x.sample_id.str.split("_").apply(lambda y: y[0]).astype(float))
        .assign(sample_num=lambda x: x.sample_id.str.split("_").apply(lambda y: y[1]).astype(float))
        .drop('sample_id', axis=1)
    )
    book_authors_df = pd.melt(pd.DataFrame.from_dict({k:pd.Series(v) for k, v in book_authors.items()}))
    book_authors_df.columns = ['author_name', 'book_id']
    print(data_df)
    print(book_authors_df)
    data_df = data_df.merge(book_authors_df, on='book_id', how='left')
    data_df = data_df.drop(['book_id','sample_num'], axis=1)
    return data_df

# Code for PyTorch model
def get_author_num(data, sample_id, authors=['Charles Dickens', 'Jane Austen','Herman Melville']):
    sample_id = int(str(sample_id).split("_")[0])
    authors_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    author_name = pd.melt(authors_df).query("value == @sample_id").variable.tolist()[0]
    return authors.index(author_name)
def stretch(arr, min_len, pad='<PAD>'):
    return arr + (min_len-len(arr))* [pad]
def get_data_nn(book_samples, book_authors, stretch_len=1100, pad='pad'):
    if stretch_len is None:
        data = [(words, get_author_num(book_authors, sample_id)) for sample_id, words in book_samples.items()]
    else:
        data = [(stretch(words, stretch_len, pad), get_author_num(book_authors, sample_id)) for sample_id, words in book_samples.items()]
    return data
def get_data_nn_auto(book_samples, stretch_len=1100, pad='pad'):
    if stretch_len is None:
        data = [(words, sample_id.split("_")[0]) for sample_id, words in book_samples.items()]
    else:
        data = [(stretch(words, stretch_len, pad), sample_id.split("_")[0]) for sample_id, words in book_samples.items()]
    return data
def get_vocab(data, vocab_size=1000):
    all_words = [word for words, _ in data for word in words]
    word_counts = Counter(all_words)
    vocab = {word: idx + 1 for idx, (word, count) in enumerate(word_counts.most_common(vocab_size-1))}
    vocab['<PAD>'] = 0
    return vocab