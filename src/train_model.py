# %%
import pandas as pd
import numpy as np
import utils
import model as md

# %%
# Load in datasets
book_contents_train = utils.load_book_contents(utils.book_authors_train)
book_contents_test = utils.load_book_contents(utils.book_authors_test)
# Tokenize the data
books_train_wtoks = utils.wtok_books(book_contents_train)
books_test_wtoks = utils.wtok_books(book_contents_test)
# Get 100 samples per book of around 1000 words each
book_samples_train = utils.get_samples(books_train_wtoks, 100, [500, 1000], random_seed=42)
book_samples_test = utils.get_samples(books_test_wtoks, 100, [500, 1000], random_seed=42)

# %%
train_data = utils.get_data_nn(book_samples_train, utils.book_authors_train, 1000)
test_data = utils.get_data_nn(book_samples_test, utils.book_authors_test, 1000)
vocab = utils.get_vocab(train_data)

# %%
model = md.NNModel()
nn_model = model.get_raw_model()

# %%
# Model, loss, optimizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

train_dataset = md.TextDataset(train_data, vocab)
test_dataset = md.TextDataset(test_data, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# %%
# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    print(epoch)
    nn_model.train()
    loss_vals = []
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        loss_vals.append(loss.item())
        optimizer.step()
    print(np.mean(loss_vals))
# %%
