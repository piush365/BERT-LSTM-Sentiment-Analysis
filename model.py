import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class LSTMRoberta(nn.Module):
    def __init__(self, num_layers=6, lstm_hidden_size=256, num_labels=2):
        super(LSTMRoberta, self).__init__()

        # Create modified RoBERTa config
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_hidden_layers = num_layers  # Reduce from 12 to 6

        self.roberta = RobertaModel(config)  # Load modified RoBERTa
        self.roberta.gradient_checkpointing_enable()

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,  # RoBERTa hidden size (768)
            hidden_size=lstm_hidden_size,  # LSTM hidden size
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )

        # Define classifier
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)  # BiLSTM doubles hidden size

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass through RoBERTa
        roberta_outputs = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = roberta_outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_size)

        # Pass through LSTM
        lstm_out, _ = self.lstm(hidden_states)  # Output shape: (batch, seq_len, hidden_size*2)
        pooled_output = lstm_out[:, -1, :]  # Take the last timestep output

        # Pass through classifier
        logits = self.classifier(pooled_output)

        # Compute loss if labels are provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits  # Return both loss and logits

        return logits  # If no labels, return only logits
