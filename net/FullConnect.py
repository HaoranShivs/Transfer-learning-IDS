import torch
import torch.nn as nn


class linear3_Relu(nn.Module):
    """
    network combined with linear
    """
    def __init__(self, input_channel, output_channel):
        super(linear3_Relu, self).__init__()
        self.fc_in = nn.Linear(input_channel, 128)
        self.fc_1 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, output_channel)
        self.softmax = nn.Softmax(-1)

        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.fc_in(input)
        x = self.relu(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x