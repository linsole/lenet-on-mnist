from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

class mnist(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        self.file = pd.read_csv(csv_file)
        self.data = torch.Tensor(np.array(self.file, ndmin=2)) # convert to FloatTensor as convolution does not accept long tensor

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        label = self.data[idx][0]
        img = torch.unsqueeze(self.data[idx][1:].view((28, -1)), 0) # add a channel dimension

        return img, label

if __name__ == "__main__":
    dataset = mnist(os.path.join("MNIST", "csv", "mnist_train.csv"))
    dataloader = DataLoader(dataset, batch_size=8)

    for batch, (X, y) in enumerate(dataloader):
        # for img in batches:
        print(f"batch: {batch}")
        print(f"X: {X.size()}")
        print(f"y: {y}")
        break
        # plt.imshow(X.permute(1, 2, 0), cmap='gray')
        # plt.show()