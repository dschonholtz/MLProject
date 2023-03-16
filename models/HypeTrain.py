"""
This is a transformer model to see if they really are better at everything lol.
"""
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn

input_dim = 1024 # Input dimension of the numpy tensor
output_dim = 2 # Output dimension of the classification task
nhead = 4 # Number of attention heads
num_layers = 4 # Number of transformer layers
dropout = 0.4 # Dropout rate


# Define the transformer classifier
class NaiveTransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer1 = nn.TransformerEncoderLayer(1024, nhead, dropout=dropout)
        self.transformer1 = nn.TransformerEncoder(encoder_layer1, 6)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.transformer1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, reduced_seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(reduced_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        positional_enc = torch.zeros(reduced_seq_len, 1, d_model)
        positional_enc[:, 0, 0::2] = torch.sin(position * div_term)
        positional_enc[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_enc', positional_enc)

    def forward(self, x):
        x = x + self.positional_enc[:x.size(1), :]
        return self.dropout(x)


class Conv1DModelWithTransformers(nn.Module):
    def __init__(self):
        super(Conv1DModelWithTransformers, self).__init__()
        self.batch_norm = nn.BatchNorm1d(1)
        self.conv1 = nn.Conv1d(1, 32, 10)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 64, 10)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(64, 128, 10)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(128, 256, 10)
        self.pool4 = nn.MaxPool1d(4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=64, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        self.fc3 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        x = self.sigmoid(self.fc3(x))
        return x


def main():

    # Create a model instance
    # model = CustomTransformerModel(num_layers, input_size, d_model, nhead, dim_feedforward, dropout, num_classes)

    model = NaiveTransformerClassifier()
    input_tensor = torch.rand(32, 1, 1024)  # (batch size, sequence length, d_model)
    output = model(input_tensor)
    print(output.shape)  # Should output (32, num_classes)

    # use a learning rate finder to find a good learning rate


    # print model info:
    print(model)

    # Calculate the total number of parameters
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_parameters)


if __name__ == '__main__':
    main()

