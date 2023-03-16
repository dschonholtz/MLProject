import torch
import torch.nn as nn
import torch.nn.functional as F


class AlisModel(nn.Module):
    def __init__(self):
        super(AlisModel, self).__init__()
        self.batch_norm = nn.BatchNorm1d(1)
        self.conv1 = nn.Conv1d(1, 32, 10)
        self.conv2 = nn.Conv1d(32, 64, 10)
        self.conv3 = nn.Conv1d(64, 128, 10)
        self.conv4 = nn.Conv1d(128, 256, 10)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Input: (batch_size, 1, 1024)
        x = self.batch_norm(x)
        # Output: (batch_size, 1, 1024)

        x = F.relu(self.conv1(x))
        # Output: (batch_size, 32, 1015)

        x = F.max_pool1d(x, 4)
        # Output: (batch_size, 32, 253)

        x = F.relu(self.conv2(x))
        # Output: (batch_size, 64, 244)

        x = F.max_pool1d(x, 4)
        # Output: (batch_size, 64, 61)

        x = F.relu(self.conv3(x))
        # Output: (batch_size, 128, 52)

        x = F.max_pool1d(x, 4)
        # Output: (batch_size, 128, 13)

        x = F.relu(self.conv4(x))
        # Output: (batch_size, 256, 4)

        x = F.max_pool1d(x, 4)
        # Output: (batch_size, 256, 1)

        x = x.view(x.size(0), -1)
        # Output: (batch_size, 256)

        x = F.relu(self.fc1(x))
        # Output: (batch_size, 64)

        x = self.dropout(x)
        # Output: (batch_size, 64)

        x = F.relu(self.fc2(x))
        # Output: (batch_size, 64)

        x = torch.sigmoid(self.fc3(x))
        # Output: (batch_size, 2)

        return x


def main():
    model = AlisModel()
    input_tensor = torch.rand(32, 1, 1024)  # (batch size, channels, sequence length)
    output = model(input_tensor)
    print(output.shape)  # Should output (32, num_classes)

    # print model info:
    print(model)

    # Calculate the total number of parameters
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_parameters)


if __name__ == '__main__':
    main()
