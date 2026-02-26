class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0}

    def encode(self, text):
        tokens = []
        for word in text.lower().split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
            tokens.append(self.vocab[word])
        return tokens
