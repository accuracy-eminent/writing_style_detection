# %%
import pandas as pd
import numpy as np
import utils
import model as md
from sklearn import metrics
import pickle
import nltk
nltk.download('punkt')

# %%
# Load in datasets
data = pd.read_pickle('../data/data.pkl')
data = data.assign(book_id_full=lambda x: x.apply(lambda a: f'{a.author_id}_{a.book_id}', axis=1))
train_data = data.query('cond == "train"').set_index('book_id_full')
test_data = data.query('cond == "test"').set_index('book_id_full')
book_contents_train = train_data.contents.to_dict()
book_contents_test = test_data.contents.to_dict()
# Get the tokenized data
books_train_wtoks = train_data.words.to_dict()
books_test_wtoks = train_data.words.to_dict()
# Get 100 samples per book of around 1000 words each
book_samples_train = utils.get_samples(books_train_wtoks, 100, [10, 1000], random_seed=42)
book_samples_test = utils.get_samples(books_test_wtoks, 100, [10, 1000], random_seed=42)


# %%
train_data_nn = utils.get_data_nn_auto(book_samples_train, None)
test_data_nn = utils.get_data_nn_auto(book_samples_test, None)
vocab = utils.get_vocab(train_data_nn)
# Vocab is of the format: {word, number}

# %%
# Get size of training data
[len(item[0]) for item in train_data_nn]


# %%
# Save the vocab
vocab_df = (
    pd.DataFrame({k: pd.Series(v) for k, v in vocab.items()})
    .T.reset_index()
)
vocab_df.columns = ['word', 'num']
vocab_df.to_csv("../data/vocab.csv", index=False)

# %%
# Load the vocab
vocab_df = pd.read_csv("../data/vocab.csv")
vocab_loaded = pd.Series(vocab_df.num.tolist(), index=vocab_df.word.tolist()).to_dict()


# %%
# Model, loss, optimizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.manual_seed(42)

model = md.NNModel(vocab_loaded)
nn_model = model.get_raw_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

train_dataset = md.TextDataset(train_data_nn, vocab_loaded)
test_dataset = md.TextDataset(test_data_nn, vocab_loaded)


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(epoch)
    nn_model.train()
    loss_vals = []
    total_correct = 0
    total_samples = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        loss_vals.append(loss.item())
        _, pred_labels = torch.max(outputs, 1)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += labels.size(0)
        optimizer.step()
    print("Loss:", np.mean(loss_vals))
    print("Accuracy: ", 100 * total_correct / total_samples)

# %%
# Evaluation
nn_model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        #print(inputs)
        #print(inputs.shape)
        #print(labels)
        #print(labels.shape)
        outputs = nn_model(inputs)
        _, predicted = torch.max(outputs, 1)
        print("---")
        print(predicted)
        print(labels)
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
