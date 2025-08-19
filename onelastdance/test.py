import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def compute_accuracy(preds, labels):
    return (preds == labels).sum().item() / len(labels)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load test dataset
    dataset = load_dataset("imdb")
    test_set = dataset["test"].select(range(5000))

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    test_set = test_set.map(tokenize_function, batched=True)
    test_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataloader = DataLoader(test_set, batch_size=16)

    # Load model config and create model (using your modified class)
    config = RobertaConfig.from_pretrained("roberta-base", num_labels=2)
    model = RobertaForSequenceClassification(config)
    model.load_state_dict(torch.load("roberta_lstm_best.pth", map_location=device))
    model.to(device)
    model.eval()

    # Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = compute_accuracy(np.array(all_preds), np.array(all_labels))
    print(f"\nTest Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    test()
