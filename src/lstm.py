import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# =========================
# 1. DATASET (LIST OF SENTENCES)
# =========================

class SentenceCharDataset:
    def __init__(self, sentences, seq_length = 30):
        self.seq_length = seq_length

        # Filter very short sentences
        self.sentences = [s for s in sentences if len(s) > seq_length + 1]

        # Build vocabulary
        all_text = "".join(self.sentences)
        chars = sorted(list(set(all_text)))

        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(chars)

        # Encode sentences
        self.encoded_sentences = [
            [self.char2idx[c] for c in s]
            for s in self.sentences
        ]

    def get_batch(self, batch_size=32):
        inputs = []
        targets = []

        for _ in range(batch_size):
            sent = random.choice(self.encoded_sentences)

            start = np.random.randint(0, len(sent) - self.seq_length - 1)

            seq = sent[start:start + self.seq_length]
            target = sent[start + 1:start + self.seq_length + 1]

            inputs.append(seq)
            targets.append(target)

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long)
        )


# =========================
# 2. MODEL (same as before)
# =========================

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=512, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm  = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


# =========================
# 3. TRAINING
# =========================

def my_train(model, dataset, epochs=10, batch_size=32, lr=0.003, device="cpu"):
    # model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for _ in range(200):
            x, y = dataset.get_batch(batch_size)
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            output, _ = model(x)

            loss = criterion(
                output.reshape(-1, dataset.vocab_size),
                y.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/200:.4f}")


# =========================
# 4. STATEFUL INFERENCE
# =========================

class Predictor:
    def __init__(self, model, dataset, device="cpu"):
        # self.model = model.to(device)
        self.model = model
        self.dataset = dataset
        self.device = device
        self.hidden = None

    def reset(self):
        self.hidden = None

    def predict_next(self, char, temperature=1.0):
        self.model.eval()

        idx = torch.tensor([[self.dataset.char2idx[char]]]).to(self.device)

        with torch.no_grad():
            output, self.hidden = self.model(idx, self.hidden)

        logits = output[0, -1] / temperature
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        next_idx = np.random.choice(len(probs), p=probs)
        return self.dataset.idx2char[next_idx]
    


# =========================
# 5. USAGE EXAMPLE
# =========================

if __name__ == "__main__":
    sentences = [
        "Commander: Initiating launch sequence.",
        "Control: Copy that. All systems nominal.",
        "Командир: Запуск начат.",
        "控制: 系统一切正常。",
        "Navigation: Adjusting trajectory.",
        "Engineer: Fuel levels stable."
    ]

    dataset = SentenceCharDataset(sentences, seq_length=10)

    model = CharLSTM(dataset.vocab_size)

    # Train
    my_train(model, dataset, epochs=5)

    # Inference
    predictor = Predictor(model, dataset)

    seed = "Commander: "
    predictor.reset()

    print("Input:", seed)

    # Warm up state
    for ch in seed:
        predictor.predict_next(ch)

    generated = seed
    for _ in range(100):
        next_char = predictor.predict_next(generated[-1])
        generated += next_char

    print("Generated:\n", generated)