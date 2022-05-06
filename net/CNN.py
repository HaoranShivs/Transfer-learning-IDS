import torch
import torch.nn as nn

from torch import permute


class CNN(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(CNN, self).__init__()
        self.fc_in = nn.Linear(input_channel, 128)
        self.cnn1 = nn.Conv1d(128, 128, 1, 1)
        self.cnn2 = nn.Conv1d(128, 128, 1, 1)
        self.fc_out = nn.Linear(128, output_channel)
        self.softmax = nn.Softmax(-1)

    def forward(self, input):
        x = self.fc_in(input)
        x = permute(x, (1,0))
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = permute(x, (1,0))
        x = self.fc_out(x)
        x = self.softmax(x)
        return x


class CNN_2_linear(nn.Module):
    """
    same with CNN, but contains two layers to classification
    """
    def __init__(self, input_channel, output_channel):
        super(CNN_2_linear, self).__init__()
        self.fc_in = nn.Linear(input_channel, 128)
        self.cnn1 = nn.Conv1d(128, 128, 1, 1)
        self.cnn2 = nn.Conv1d(128, 128, 1, 1)
        self.fc_1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(64, output_channel)
        self.softmax = nn.Softmax(-1)

    def forward(self, input):
        x = self.fc_in(input)
        x = permute(x, (1,0))
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = permute(x, (1,0))
        x = self.fc_1(x)
        # x = self.relu(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x
