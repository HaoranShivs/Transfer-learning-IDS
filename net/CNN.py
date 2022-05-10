import torch
import torch.nn as nn

from torch import permute, unsqueeze, reshape


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


class CNN3_2(nn.Module):
    def __init__(self, input_shape, output_channel):
        super(CNN3_2, self).__init__()
        H, W = input_shape
        self.cnn1_in = nn.Conv2d(1, 1, 3, 1)    # (N, 1, H-2, W-2)
        self.cnn2 = nn.Conv2d(1, 1, 3, 1)   #(N, 1, H-4, W-4)
        # self.cnn3 = nn.Conv2d(1, 1, 3, 1)   #(N, 1, H-8, W-8)
        # self.fc1 = nn.Linear((H-8)*(W-8), 128)
        self.fc1 = nn.Linear((H-4)*(W-4), 128)
        self.relu = nn.ReLU()
        self.fc2_out = nn.Linear(128, output_channel) 
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs):
        x = unsqueeze(inputs, 1)
        x = self.cnn1_in(x)
        x = self.cnn2(x)
        # x = self.cnn3(x)
        x = reshape(x,(x.shape[0], -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2_out(x)
        x = self.softmax(x)
        return x


class CNN3_3(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(CNN3_3, self).__init__()
        self.fc1_in = nn.Linear(input_channel, 128) # (N, 128)
        self.cnn1 = nn.Conv2d(1, 1, 5, 1)    # (N, 1, 4, 12)
        self.cnn2 = nn.Conv2d(1, 1, 3, 1)   #(N, 1, 2, 10)
        # self.cnn3 = nn.Conv2d(1, 1, 3, 1)   #(N, 1, H-8, W-8)
        # self.fc1 = nn.Linear((H-8)*(W-8), 128)
        self.fc2 = nn.Linear(2*10, 128)
        self.relu = nn.ReLU()
        self.fc3_out = nn.Linear(128, output_channel) 
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs):
        x = self.fc1_in(inputs)
        x = unsqueeze(x, 1)
        x = reshape(x, (x.shape[0], 1, 8, 16))
        x = self.cnn1(x)
        x = self.cnn2(x)
        # x = self.cnn3(x)
        x = reshape(x,(x.shape[0], -1))
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3_out(x)
        x = self.softmax(x)
        return x


class CNN3(nn.Module):
    def __init__(self, input_shape, output_channel):
        super(CNN3, self).__init__()
        H, W = input_shape
        self.cnn1_in = nn.Conv2d(1, 1, 5, 1)    # (N, 1, H-4, W-4)
        self.cnn2 = nn.Conv2d(1, 1, 3, 1)   #(N, 1, H-6, W-6)
        # self.cnn3 = nn.Conv2d(1, 1, 3, 1)   #(N, 1, H-8, W-8)
        # self.fc1 = nn.Linear((H-8)*(W-8), 128)
        self.fc1 = nn.Linear((H-6)*(W-6), 128)
        self.relu = nn.ReLU()
        self.fc2_out = nn.Linear(128, output_channel) 
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs):
        x = unsqueeze(inputs, 1)
        x = self.cnn1_in(x)
        x = self.cnn2(x)
        # x = self.cnn3(x)
        x = reshape(x,(x.shape[0], -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2_out(x)
        x = self.softmax(x)
        return x


class CNN_2_linear_Relu(nn.Module):
    """
    same with CNN, but contains two layers to classification
    """
    def __init__(self, input_channel, output_channel):
        super(CNN_2_linear_Relu, self).__init__()
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
        x = self.relu(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    print(CNN(2,3))
    print(CNN_2_linear_Relu(2,3))
