import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.utils import get_output_tensor
from utils.utils import get_psd_tensor
from utils.utils import max_step


class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6,16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_simple_net_model(filename):
    model = SimpleNet()
    model.load_state_dict(torch.load(filename))
    model.eval()

    return model

def max_batch_num(raw, batch_size=1000):
    return int(max_step(raw)/batch_size)


def get_batch(raw, batch_num, batch_size=1000, subsample=10):
    begin_time = batch_num*batch_size
    end_time = 0

    if batch_num + 1 >= max_batch_num(raw, batch_size):
        end_time = int(begin_time + max_step(raw)%batch_size)
    else:
        end_time = int(begin_time + batch_size)

    tensor_len = len(range(begin_time, end_time,subsample))
    input = torch.randn(tensor_len, 1, 32, 33)
    output = torch.randn(tensor_len, 6)

    iterator = 0
    for i in range(begin_time, end_time, subsample):
        input[iterator] = get_psd_tensor(raw, i)
        output[iterator] = get_output_tensor(raw, i)
        iterator = iterator + 1

    return input, output


def train_net(net, raw, batch_size=120, out_dir="./SimpleNet/Models"):
    print("Train Net Function")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    batch_num = max_batch_num(raw, batch_size=batch_size)
    print("Num of batches per epoch:", batch_num)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):

        running_loss = 0.0
        for i in range(batch_num):
            inputs, outputs_true = get_batch(raw, i, batch_size=batch_size, subsample=15)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, outputs_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('Epoch %d; Batch: %d; loss: %.3f' %
                      (epoch+1, i +1, running_loss*10))
                running_loss = 0.0

        path = out_dir + 'sn-epoch-' +str(epoch) + '.pth'
        torch.save(net.state_dict(), path)
