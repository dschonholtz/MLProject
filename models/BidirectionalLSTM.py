"""
This is the 
"""
# Import modules
import torch
import torch.nn as nn
import numpy as np
import torchvision.ops


# Define the bidirectional LSTM classifier
class BiLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        dropout = 0.2
        self.lstm1 = nn.LSTM(1024, 256, 1, dropout=dropout, bidirectional=True)
        self.lstm2 = nn.LSTM(256 * 2, 64, 1, dropout=dropout, bidirectional=True)
        self.lstm3 = nn.LSTM(64 * 2, 32, 1,  dropout=dropout, bidirectional=True)
        self.linear1 = nn.Linear(32 * 2, 16)
        self.linear2 = nn.Linear(32, 8)
        self.linear3 = nn.Linear(8, 2)

    def forward(self, x):
        # Convert the numpy tensor to a PyTorch tensor
        x = torch.from_numpy(x).float()
        # Add a dummy dimension for the batch size and transpose the input to match the expected shape of (seq_len, batch, input_size)
        x = x.unsqueeze(0).transpose(0, 1)
        # Pass the input through the bidirectional LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        # Remove the dummy dimensions
        x = x.squeeze(0).squeeze(0)
        # Pass the output through the linear layer

        # Take the last output of both directions and concatenate them
        x = torch.cat((x[-1, :, :self.hidden_dim], x[0, :, self.hidden_dim:]), dim=1)
        # Pass the output through the linear layer
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Conv1DModelWithLSTM(nn.Module):
    def __init__(self):
        super(Conv1DModelWithLSTM, self).__init__()
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

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)

        self.fc3 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

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
        x, _ = self.lstm(x)
        x = x.squeeze(1)

        x = self.sigmoid(self.fc3(x))
        return x


def main():
    model = Conv1DModelWithLSTM()
    input_tensor = torch.rand(32, 1, 1024)  # (batch size, sequence length, d_model)
    output = model(input_tensor)
    print(output.shape)  # Should output (32, num_classes)

    # print model info:
    print(model)

    # Calculate the total number of parameters
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_parameters)


if __name__ == '__main__':
    main()
