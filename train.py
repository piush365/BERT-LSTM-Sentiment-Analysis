# import torch
# import torch.nn as nn
# import torch.optim as optim
# from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import numpy as np
# from torch.amp import autocast, GradScaler

# # Function to convert sentiment to label
# def encode_label(sentiment):
#     return 1 if sentiment == "positive" else 0

# # Custom Dataset class for IMDb
# class IMDbDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer):
#         self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=256)
#         self.labels = list(labels)

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['label'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# # Xavier weight initialization
# def init_weights(module):
#     if isinstance(module, nn.Linear):
#         nn.init.xavier_uniform_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.LSTM):
#         for name, param in module.named_parameters():
#             if 'weight_ih' in name or 'weight_hh' in name:
#                 nn.init.xavier_uniform_(param)
#             elif 'bias' in name:
#                 nn.init.zeros_(param)

# # Accuracy computation
# def compute_accuracy(preds, labels):
#     return (preds == labels).sum().item() / len(labels)

# # Load and prepare small data
# def prepare_data(tokenizer):
#     df = pd.read_csv("IMDB Dataset.csv", on_bad_lines='skip')
#     df = df.sample(n=10000, random_state=42).reset_index(drop=True)
#     df['label'] = df['sentiment'].apply(encode_label)

#     train_texts, temp_texts, train_labels, temp_labels = train_test_split(
#         df['review'], df['label'], test_size=0.3, random_state=42)
#     val_texts, test_texts, val_labels, test_labels = train_test_split(
#         temp_texts, temp_labels, test_size=0.5, random_state=42)

#     train_dataset = IMDbDataset(train_texts, train_labels, tokenizer)
#     val_dataset = IMDbDataset(val_texts, val_labels, tokenizer)
#     test_dataset = IMDbDataset(test_texts, test_labels, tokenizer)

#     return train_dataset, val_dataset, test_dataset

# # Training with validation and testing after each epoch
# def train():
#     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#     train_data, val_data, test_data = prepare_data(tokenizer)

#     train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
#     val_loader = DataLoader(val_data, batch_size=8, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_data, batch_size=8, num_workers=2, pin_memory=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     config = RobertaConfig.from_pretrained(
#         "roberta-base",
#         hidden_dropout_prob=0.5,
#         attention_probs_dropout_prob=0.5,
#         num_labels=2
#     )

#     model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)
#     model.apply(init_weights)
#     model.to(device)

#     optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)

#     # 👇 AMP scaler
#     scaler = GradScaler()

#     for epoch in range(15):
#         print(f"\nEpoch {epoch + 1}")
#         model.train()
#         total_loss = 0.0
#         running_preds, running_labels = [], []

#         for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["label"].to(device)

#             optimizer.zero_grad()

#             # 👇 Mixed precision forward pass
#             with autocast(device_type='cuda'):
#                 outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#                 loss = outputs.loss
#                 logits = outputs.logits

#             # 👇 Scaled backward + optimizer step
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             total_loss += loss.item()
#             preds = torch.argmax(logits, dim=-1)
#             running_preds.extend(preds.cpu().numpy())
#             running_labels.extend(labels.cpu().numpy())

#         train_acc = compute_accuracy(np.array(running_preds), np.array(running_labels))
#         print(f"Epoch {epoch + 1} done. Train Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")

#         # Validation
#         model.eval()
#         val_preds, val_labels = [], []
#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 labels = batch["label"].to(device)

#                 outputs = model(input_ids, attention_mask=attention_mask)
#                 logits = outputs.logits
#                 preds = torch.argmax(logits, dim=-1)

#                 val_preds.extend(preds.cpu().numpy())
#                 val_labels.extend(labels.cpu().numpy())

#         val_acc = compute_accuracy(np.array(val_preds), np.array(val_labels))
#         print(f"Validation Accuracy after Epoch {epoch + 1}: {val_acc:.4f}")

#         # Test
#         test_preds, test_labels = [], []
#         with torch.no_grad():
#             for batch in test_loader:
#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 labels = batch["label"].to(device)

#                 outputs = model(input_ids, attention_mask=attention_mask)
#                 logits = outputs.logits
#                 preds = torch.argmax(logits, dim=-1)

#                 test_preds.extend(preds.cpu().numpy())
#                 test_labels.extend(labels.cpu().numpy())

#         test_acc = compute_accuracy(np.array(test_preds), np.array(test_labels))
#         print(f"Test Accuracy after Epoch {epoch + 1}: {test_acc:.4f}")

#         # Save model
#         torch.save(model.state_dict(), f"roberta_epoch{epoch+1}.pth")
#         print(f"Model saved as roberta_epoch{epoch+1}.pth")

# # Start training
# if __name__ == "__main__":
#     train()

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
import random
import torch.nn as nn

# Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
# df = pd.read_csv("IMDB Dataset.csv", on_bad_lines='skip')
# df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# # Split dataset: 25K train + 25K test
# train_df = df.iloc[:50000].reset_index(drop=True)
# test_df = df.iloc[25000:].reset_index(drop=True)

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("IMDB Dataset.csv", on_bad_lines='skip')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train-test split: 90% train, 10% test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['sentiment'])

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Datasets and Dataloaders
train_dataset = SentimentDataset(train_df['review'].tolist(), train_df['sentiment'].tolist(), tokenizer)
test_dataset = SentimentDataset(test_df['review'].tolist(), test_df['sentiment'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)

# Freeze first 6 layers
for i in range(6):
    for param in model.roberta.encoder.layer[i].parameters():
        param.requires_grad = False

# Optimizer and Scheduler
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
num_epochs = 5
grad_accum_steps = 2
scaler = torch.amp.GradScaler("cuda")

scheduler = OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10,
    final_div_factor=100
)

# Training
# Load saved model weights
model.load_state_dict(torch.load("roberta_imdb_trained.pt"))

model.train()
for epoch in range(num_epochs):
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for step, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            logits = outputs.logits

        scaler.scale(loss).backward()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * grad_accum_steps

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        loop.set_postfix(loss=loss.item())

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

# Save model
torch.save(model.state_dict(), "roberta_imdb_trained.pt")

# ✅ Evaluation on test set
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")
