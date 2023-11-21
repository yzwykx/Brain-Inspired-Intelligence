import os
import time
import argparse
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
import torchvision
import numpy as np

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer


class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            )

    def forward(self, x: torch.Tensor):
        return self.layer(x)

# 参数设置
net = SNN(tau=2.0)
net.to('cuda:0')

# 数据加载器初始化
train_dataset = torchvision.datasets.MNIST(
        root="./mnist/",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
)
test_dataset = torchvision.datasets.MNIST(
        root="./mnist/",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
)

train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
)
test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
)

scaler = None
start_epoch = 0
max_test_acc = -1
epochs = 10
T = 100
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

encoder = encoding.PoissonEncoder()

for epoch in range(start_epoch, epochs):
    start_time = time.time()
    net.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    for img, label in train_data_loader:
        optimizer.zero_grad()
        img = img.to('cuda:0')
        label = label.to('cuda:0')
        label_onehot = F.one_hot(label, 10).float()
        out_fr = 0.
        for t in range(T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_fr = out_fr / T
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()
        optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(net)

    train_time = time.time()
    train_speed = train_samples / (train_time - start_time)
    train_loss /= train_samples
    train_acc /= train_samples
    print('train_loss', train_loss, epoch)
    print('train_acc', train_acc, epoch)

net.eval()
test_loss = 0
test_acc = 0
test_samples = 0
with torch.no_grad():
    for img, label in test_data_loader:
        img = img.to('cuda:0')
        label = label.to('cuda:0')
        label_onehot = F.one_hot(label, 10).float()
        out_fr = 0.
        for t in range(T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_fr = out_fr / T
        loss = F.mse_loss(out_fr, label_onehot)

        test_samples += label.numel()
        test_loss += loss.item() * label.numel()
        test_acc += (out_fr.argmax(1) == label).float().sum().item()
        functional.reset_net(net)
test_time = time.time()
test_speed = test_samples / (test_time - train_time)
test_loss /= test_samples
test_acc /= test_samples
print('test_loss', test_loss, epoch)
print('test_acc', test_acc, epoch)
print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')