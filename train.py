import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataprocess.kdd_cup99 import KDD_CUP_99_DataLoader
from net.CNN_KDD import CNN_KDD

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # fundamental setting
    data_root_dir = 'E:/DataSets/kddcup.data'
    batch_size = 256
    epoch = 100

    dataset = KDD_CUP_99_DataLoader(data_root_dir, batch_size)
    net = CNN_KDD(41,23,3).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-2)
    

