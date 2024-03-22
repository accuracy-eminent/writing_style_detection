# %%
import pandas as pd
import numpy as np
import utils
import model as md
from sklearn import metrics
import pickle

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
num_epochs = 10
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
# Evaluation
nn_model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = nn_model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# %%
# Save model weights
torch.save(nn_model.state_dict, "../data/nn.pth")


# %%
# Tabular model
# Load in data
# Do feature engineering
# Use ngram frequency as features
# cd_1grams is the frequency of 1-grams associated with Charles Dickens, for example
data_df_train = utils.get_data_df(book_samples_train, utils.book_authors_train)
data_df_test = utils.get_data_df(book_samples_test, utils.book_authors_test)

# %%
# Prepare datasets for training
tgt_cols = data_df_test.columns
tgt_cols = ['author_name']
X_train = data_df_train.drop(tgt_cols,axis=1)
y_train = data_df_train.filter(tgt_cols).to_numpy().ravel()
X_test = data_df_test.drop(tgt_cols,axis=1)
y_test = data_df_test.filter(tgt_cols).to_numpy().ravel()

# %%
model_tab = md.TabularModel()
tb_model = model_tab.get_raw_model()
# %%
tb_model.fit(X_train, y_train)
y_pred = tb_model.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# %%
# Save model
with open( "../data/tabular.pkl", "wb") as file:
    pickle.dump(tb_model, file)
# %%
