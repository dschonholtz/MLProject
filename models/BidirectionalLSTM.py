"""
This is the 
"""
# Import modules
import torch
import torch.nn as nn
import numpy as np

# Define hyperparameters
input_dim = 1024 # Input dimension of the numpy tensor
output_dim = 2 # Output dimension of the classification task
hidden_dim = 256 # Hidden dimension of the LSTM
num_layers = 5 # Number of LSTM layers
dropout = 0.4 # Dropout rate


# Define the bidirectional LSTM classifier
class BiLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        self.linear1 = nn.Linear(hidden_dim * 2, 32)
        self.linear2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # Convert the numpy tensor to a PyTorch tensor
        x = torch.from_numpy(x).float()
        # Add a dummy dimension for the batch size and transpose the input to match the expected shape of (seq_len, batch, input_size)
        x = x.unsqueeze(0).transpose(0, 1)
        # Pass the input through the bidirectional LSTM
        x, _ = self.lstm(x)
        # Take the last output of both directions and concatenate them
        x = torch.cat((x[-1, :, :hidden_dim], x[0, :, hidden_dim:]), dim=1)
        # Pass the output through the linear layer
        x = self.linear1(x)
        x = self.linear2(x)
        return x
