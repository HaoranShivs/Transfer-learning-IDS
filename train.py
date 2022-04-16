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

    writer1 = SummaryWriter('log/exp')

    for t in range(epoch):
        running_loss = 0
        for step, (x, y) in enumerate(dataset):
            x = x.to(device)

            y = F.one_hot(y.long(), num_classes=23).float()
            y = y.to(device)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad() 

            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            # visualize loss
            running_loss += loss.item()

        if t % 10 == 9:
            writer1.add_scalar('training loss', running_loss, t)
            checkpoint = {"net": net.state_dict(), 'optimizer':optimizer.state_dict(), "epoch": t}
            if not os.path.isdir("/Transfer-learning-IDS/history/CNN_KDD/checkpoint"):
                os.makedirs("/Transfer-learning-IDS/history/CNN_KDD/checkpoint")
            torch.save(checkpoint, '/Transfer-learning-IDS/history/CNN_KDD/checkpoint/ckpt_best_%s.pth' %(str(t)))
    writer1.close()


    

