# This file trains the model. Apart from other details, for now we focus on 
# 4 major parts: dataloader, model, loss function and optimizer and use this 
# as a backbone line during studying.

import torch
from dataset import mnist
import os
from torch.utils.data import DataLoader
import yaml
from model import LeNet
import torch.nn as nn
import time

torch.manual_seed(42)

# prepare dataloader
train_dataset = mnist(os.path.join("MNIST", "csv", "mnist_train.csv"))
test_dataset = mnist(os.path.join("MNIST", "csv", "mnist_test.csv"))
train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(train_dataset, batch_size=32)

# prepare network model
with open("config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
lenet = LeNet(config)
# lenet.load_state_dict(torch.load("saves.pth"))

# prepare loss function
loss_fn = nn.CrossEntropyLoss()

# prepare optimizer
optimizer = torch.optim.SGD(lenet.parameters(), lr=1e-3)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train() # I suppose this is necessary because i used batch norm layer
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        # not sure why y has to be long tensor? Plus it seems that the pred don't need to be changed.
        loss = loss_fn(pred, y.type(torch.LongTensor))

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval() # I suppose this is necessary because i used batch norm layer
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y.type(torch.LongTensor)).item() # same problem here.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 20
t0 = time.time()
for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train_loop(train_dataloader, lenet, loss_fn, optimizer)
    test_loop(test_dataloader, lenet, loss_fn)

torch.save(lenet.state_dict(), "saves.pth")
print(f"Done. Average: {(time.time()-t0) / epochs} per epoch.")