import os
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW 

# ======= Dataset & Preprocessing =======
class PoliticalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# ======= Load & Prepare Data =======
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "../data")
    train_df = pd.read_json(os.path.join(data_path, "train.json"), lines=True)
    dev_df = pd.read_json(os.path.join(data_path, "dev.json"), lines=True)
    df = pd.concat([train_df, dev_df]).dropna()
    df = df[['sentence', 'polarity']].rename(columns={'sentence': 'text', 'polarity': 'label'})
    df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})
    return df

# ======= Training =======
def train_model(model, tokenizer, train_texts, train_labels, epochs=3):
    dataset = PoliticalDataset(train_texts, train_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} — Loss: {total_loss:.4f}")

# ======= Export Inference Wrapper =======
class PoliticalSentimentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, texts):
        self.model.eval()
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            preds = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1).tolist()
        return [self.label_map[p] for p in preds]

def save_model(model, tokenizer):
    sentiment_model = PoliticalSentimentModel(model, tokenizer)
    with open("political_sentiment_model.pkl", "wb") as f:
        pickle.dump(sentiment_model, f)

# ======= Main =======
if __name__ == "__main__":
    df = load_dataset()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    train_model(model, tokenizer, df['text'].tolist(), df['label'].tolist(), epochs=3)
    save_model(model, tokenizer)
