class FakeNewsDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.data = df['combined_text'].tolist()
        self.labels = df['label'].tolist()
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        vectorized_text = self.vectorizer.transform([text]).toarray().squeeze()
        return torch.tensor(vectorized_text, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
