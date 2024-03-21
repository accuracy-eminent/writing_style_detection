from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class AuthorClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(AuthorClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])  # Take the last LSTM output for each sequence
        return output

class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, label = self.data[idx]
        word_indices = [self.vocab.get(word, 0) for word in words]
        return torch.tensor(word_indices), torch.tensor(int(label))  # Assuming labels are integers

class Model():
    def __init__(self):
        pass
    def get_raw_model(self):
        return self.model
    def set_raw_model(self, model):
        self.model = model
    def train(self, train_data, test_data):
        # Data will be in the format {'authornum_bookid': [words]}
        pass
    def load_weights(self, file):
        pass
    def save_weights(self, file):
        pass
    def _load_data(self, text):
        pass
    def predict(self):
        pass

class TabularModel(Model):
    def __init__(self):
        self.model = GradientBoostingClassifier()
    def train(self, train_data, test_data):
        pass
    def load_weights(self, file):
        self.model = pickle.load(file)
    def _load_data(self, data):
        out_data = utils.get_data_df([data], ['unknown'])
        return out_data
    def predict(self, data):
        y_pred = self.model.predict_proba(self.data)
        return y_pred

class NNModel(Model):
    def __init__(self, embedding_dim=64, hidden_size=128, num_classes=3, vocab_size=1000):
        # TODO: define vocab
        # self.vocab = load vocab from file
        embedding_dim = 64
        hidden_size = 128
        num_classes = 3
        vocab_size = 1000
        self.model = AuthorClassifier(vocab_size, embedding_dim, hidden_size, num_classes)
    def load_weights(self):
        self.model(model.load_state_dict(torch.load("../data/weights_new.pth")()))
    def _load_data(self, data):
        out_data = TextDataset(data, self.vocab)
        return out_data
    def predict(self):
        y_pred = model(self.data)
        return y_pred