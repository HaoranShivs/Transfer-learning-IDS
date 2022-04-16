import torch
import torch.nn as nn

from torch import permute


class CNN_KDD(nn.Module):
    def __init__(self, input_channel, output_channel, conv1d_num=2):
        super(CNN_KDD, self).__init__()
        self.conv1d_num = conv1d_num
        self.fc_in = nn.Linear(input_channel, 128)
        self.cnn1 = nn.Conv1d(128, 128, 1, 1)
        self.cnn2 = nn.Conv1d(128, 128, 1, 1)
        # self.cnn3 = nn.Conv1d(128, 128, 1, 1)
        # self.cnn4 = nn.Conv1d(128, 128, 1, 1)
        self.fc_out = nn.Linear(128, output_channel)
        self.softmax = nn.Softmax(-1)

    def forward(self, input):
        x = self.fc_in(input)
        x = permute(x, (1,0))
        x = self.cnn1(x)
        x = self.cnn2(x)
        # x = self.cnn3(x)
        # x = self.cnn4(x)
        x = permute(x, (1,0))
        x = self.fc_out(x)
        x = self.softmax(x)
        return x
