from sklearn.ensemble import GradientBoostingClassifier

class Model():
    def __init__(self):
        pass
    def train(self, training_data):
        pass
    def load_weights(self, file):
        pass
    def load_data(self, text):
        pass
    def predict(self):
        pass

class TabularModel(Model):
    def __init__(self):
        self.model = GradientBoostingClassifier()
    def load_weights(self, file):
        self.model = pickle.load(file)
    def load_data(self, data):
        self.data = utils.get_data_df([data], ['unknown'])
    def predict(self):
        y_pred = self.model.predict_proba(self.data)
        return y_pred

class NNModel(Model):
    def __init__(self):
        # TODO: define vocab
        # self.vocab = load vocab from file
        embedding_dim = 64
        hidden_size = 128
        num_classes = 3
        vocab_size = 1000
        self.model = AuthorClassifier(vocab_size, embedding_dim, hidden_size, num_classes)
    def load_weights(self):
        pass
    def load_data(self, data):
        self.data = TextDataset(data, self.vocab)
        pass
    def predict(self):
        y_pred = model(self.data)
        return y_pred