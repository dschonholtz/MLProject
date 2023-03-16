"""
This is a 1d CNN replication effort of the existing 1d CNN that the lab uses
The format is as follows:
Input:1x1024
4 Conv layers into max pool with leaky relu
2x 32 dense layers
Sigmoid focal cross entropy
"""
import torch.nn as nn
import torch


class OneDCNN(nn.Module):
    def __init__(self):
        super(OneDCNN, self).__init__()
        self.conv1d_16 = nn.Conv1d(in_channels=1, out_channels=9, kernel_size=5)
        # 1020
        self.leaky_relu_18 = nn.LeakyReLU(0.1)
        self.conv1d_17 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=5)
        # 1016
        self.leaky_relu_19 = nn.LeakyReLU(0.1)
        self.max_pooling1d_8 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 508
        self.conv1d_18 = nn.Conv1d(in_channels=9, out_channels=13, kernel_size=15)
        # 508 - 15 + 1 = 494
        self.leaky_relu_20 = nn.LeakyReLU(0.1)
        self.conv1d_19 = nn.Conv1d(in_channels=13, out_channels=13, kernel_size=15)
        # 494 - 15 + 1 = 480
        self.leaky_relu_21 = nn.LeakyReLU(0.1)
        self.max_pooling1d_9 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 480 / 2 = 240
        self.conv1d_20 = nn.Conv1d(in_channels=13, out_channels=21, kernel_size=25)
        # 240 - 25 + 1 = 216
        self.leaky_relu_22 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.5)
        self.conv1d_21 = nn.Conv1d(in_channels=21, out_channels=21, kernel_size=25)
        # 216 - 25 + 1 = 192
        self.leaky_relu_23 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.max_pooling1d_10 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 192 / 2 = 96
        self.conv1d_22 = nn.Conv1d(in_channels=21, out_channels=21, kernel_size=35)
        # 96 - 35 + 1 = 62
        self.leaky_relu_24 = nn.LeakyReLU(0.1)
        self.dropout3 = nn.Dropout(0.5)
        self.conv1d_23 = nn.Conv1d(in_channels=21, out_channels=21, kernel_size=35)
        # 62 - 35 + 1 = 28
        self.leaky_relu_25 = nn.LeakyReLU(0.1)
        self.dropout4 = nn.Dropout(0.5)
        self.max_pooling1d_11 = nn.MaxPool1d(kernel_size=2, stride=2)
        # 28 / 2 = 14
        self.dense_1 = nn.Linear(in_features=14 * 21, out_features=16)
        self.leaky_relu_26 = nn.LeakyReLU(0.1)
        self.dropout5 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        x = self.conv1d_16(x)
        x = self.leaky_relu_18(x)
        x = self.conv1d_17(x)
        x = self.leaky_relu_19(x)
        x = self.max_pooling1d_8(x)
        x = self.conv1d_18(x)
        x = self.leaky_relu_20(x)
        x = self.conv1d_19(x)
        x = self.leaky_relu_21(x)
        x = self.max_pooling1d_9(x)
        x = self.conv1d_20(x)
        x = self.leaky_relu_22(x)
        x = self.dropout1(x)
        x = self.conv1d_21(x)
        x = self.leaky_relu_23(x)
        x = self.dropout2(x)
        x = self.max_pooling1d_10(x)
        x = self.conv1d_22(x)
        x = self.leaky_relu_24(x)
        x = self.dropout3(x)
        x = self.conv1d_23(x)
        x = self.leaky_relu_25(x)
        x = self.dropout4(x)
        x = self.max_pooling1d_11(x)
        x = x.view(-1, 14 * 21)
        x = self.dense_1(x)
        x = self.leaky_relu_26(x)
        x = self.dropout5(x)
        x = self.dense_2(x)
        return x


def main():

    # Create a model instance
    # model = CustomTransformerModel(num_layers, input_size, d_model, nhead, dim_feedforward, dropout, num_classes)

    model = OneDCNN()
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
