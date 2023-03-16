"""
This is a transformer model to see if they really are better at everything lol.
"""
import torch
import torch.nn as nn

input_dim = 1024 # Input dimension of the numpy tensor
output_dim = 2 # Output dimension of the classification task
nhead = 8 # Number of attention heads
num_layers = 6 # Number of transformer layers
d_model = 512 # Hidden dimension of the transformer
dropout = 0.4 # Dropout rate


# Define the transformer classifier
class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.linear1 = nn.Linear(d_model, 32)
        self.linear2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # Convert the numpy tensor to a PyTorch tensor
        x = torch.from_numpy(x).float()
        # Add a dummy dimension for the batch size and the sequence length
        x = x.unsqueeze(0).unsqueeze(0)
        # Pass the input through the transformer encoder
        x = self.transformer.encoder(x)
        # Remove the dummy dimensions
        x = x.squeeze(0).squeeze(0)
        # Pass the output through the linear layer
        x = self.linear1(x)
        x = self.linear2(x)
        return x