import torch
import torch.nn as nn

from torch import permute


class multCNN(nn.Module):
    def __init__(self, input_num_channel, input_kind_channel, output_channel):
        super(multCNN, self).__init__()
        # part num
        self.fc_input_num = nn.Linear(input_num_channel, 128)
        self.cnn_num_1 = nn.Conv1d(128, 64, 1, 1)
        self.cnn_num_2 = nn.Conv1d(64, 32, 1, 1)

        # part kind
        self.fc_input_kind = nn.Linear(input_kind_channel, 128)
        self.cnn_kind_1 = nn.Conv1d(128, 64, 1, 1)
        self.cnn_kind_2 = nn.Conv1d(64, 32, 1, 1)

        # concatenate two part
        self.cnn_3 = nn.Conv1d(64, 32, 1, 1)
        self.fc_output = nn.Linear(32, output_channel)

        self.softmax = nn.Softmax(-1)

    def forward(self, input):
        num_input, kind_input = input[0], input[1]
        # part num
        x_num = self.fc_input_num(num_input)
        x_num = permute(x_num, (1,0))
        x_num = self.cnn_num_1(x_num)
        x_num = self.cnn_num_2(x_num)

        # part kind
        x_kind = self.fc_input_kind(kind_input)
        x_kind = permute(x_kind, (1,0))
        x_kind = self.cnn_kind_1(x_kind)
        x_kind = self.cnn_kind_2(x_kind)

        # concatenate two part
        x = torch.cat((x_num, x_kind), dim=-2)
        x = self.cnn_3(x)
        x = permute(x, (1,0))
        x = self.fc_output(x)
        x = self.softmax(x)
        return x
