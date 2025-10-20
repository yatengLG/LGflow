# -*- coding: utf-8 -*-
# @Author  : LG

import numpy as np
import time

from LG_flow import nn, optim, Tensor
from LG_flow.utils.data import DataLoader, Mnist, to_one_hot


def train(dataloader, model, loss_fn, optimizer):
    train_loss, correct = 0., 0.

    for i, (images, labels) in enumerate(dataloader):
        images = images.reshape(images.shape[0], -1) / 255.0
        images = Tensor(images)
        labels_one_hot = Tensor(to_one_hot(labels, num_classes=10))

        pred = model.forward(images)
        loss = loss_fn.forward(pred, labels_one_hot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += np.sum(np.argmax(pred.data, 1) == labels)
        train_loss += loss.data

    return train_loss / len(dataloader), correct / len(dataloader.dataset)


def test(dataloader, model, loss_fn):
    test_loss, correct = 0., 0.

    for i, (images, labels) in enumerate(dataloader):
        images = images.reshape(images.shape[0], -1) / 255.0
        images = Tensor(images)
        labels_one_hot = Tensor(to_one_hot(labels, num_classes=10))

        pred = model.forward(images)
        loss = loss_fn.forward(pred, labels_one_hot)

        correct += np.sum(np.argmax(pred.data, 1) == labels)
        test_loss += loss.data

    return test_loss / len(dataloader), correct / len(dataloader.dataset)


# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.act1.forward(x)
        x = self.fc2.forward(x)
        x = self.act2.forward(x)
        x = self.fc3.forward(x)
        return x


model = MLP()

# 数据加载
train_dataset = Mnist(root="data/MNIST/raw", train=True)
test_dataset = Mnist(root="data/MNIST/raw", train=False)

train_dataloader = DataLoader(train_dataset, batch=16, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch=16, shuffle=False, drop_last=False)

# 损失函数与优化器
loss_fn = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 开始训练
print(f"| {'':^6s} | {'time':^15s} | {'loss':^15s} | {'acc':^15s} |")
print(f"| {'epoch':^6s} | {'train':^6s} | {'test':^6s} | {'train':^6s} | {'test':^6s} | {'train':^6s} | {'test':^6s} |")

for epoch in range(50):

    time1 = time.time()
    train_loss, train_correct = train(train_dataloader, model, loss_fn, optimizer)
    time2 = time.time()
    test_loss, test_correct = test(test_dataloader, model, loss_fn)
    time3 = time.time()

    print(f"| {epoch:6d} | {time2-time1:.4f} | {time3-time2:.4f} | {train_loss:.4f} | {test_loss:.4f} | {train_correct:.4f} | {test_correct:.4f} |")
