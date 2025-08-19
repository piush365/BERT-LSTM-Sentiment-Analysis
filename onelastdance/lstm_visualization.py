import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# 2. Load config
config = RobertaConfig.from_pretrained("roberta-base")
config.visualize = True  # this flag activates visualization inside your modified RobertaLayer

# 3. Load model architecture
model = RobertaForSequenceClassification(config)

# 4. Load trained weights (you trained on the modified source so it matches the architecture)
checkpoint_path = "roberta_lstm_best.pth"  # path to your .pth checkpoint
state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

# 5. Tokenize a sample input
sentence = "I really enjoyed this movie! It was fantastic and emotional."
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

# 6. Run forward pass with visualization
with torch.no_grad():
    outputs = model.roberta(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        visualize=True  # <- make sure this argument is used in your RobertaModel forward method
    )

# 7. Access visualization data from model.roberta (assuming RobertaModel collects it)
visual_data = outputs.visualization_outputs

# 8. Plot first 4 layer outputs
for i, layer in enumerate(visual_data):
    if i >= 4:
        break

    hidden = layer['output'][0]  # (seq_len, hidden_dim)
    sns.heatmap(hidden.cpu().numpy(), cmap='viridis')
    plt.title(f"Layer {layer['layer']} Output (LSTM-enhanced)")
    plt.xlabel("Hidden Size")
    plt.ylabel("Token Position")
    plt.tight_layout()
    plt.show()
