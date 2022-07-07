import torch
import torch.nn as nn
import yaml

class ConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(**kwargs)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(kwargs["out_channels"]) # we do *not* place batch norm before activation
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.batch_norm(self.relu(self.conv(x))))


class LeNet(nn.Module):
    def __init__(self, config_file):
        super().__init__()
        self.conv = self._create_conv(config_file)
        self.fc = self._create_fc(config_file)

    def _create_conv(self, config_file):
        convolution1 = ConvBlock(**config_file["convolution_layer"][0])
        convolution2 = ConvBlock(**config_file["convolution_layer"][1])
        return nn.Sequential(*[convolution1, convolution2])

    def _create_fc(self, config_file):
        return nn.Sequential(
            nn.Linear(**config_file["fully_connected_layer"][0]), 
            nn.ReLU(), 
            nn.Linear(**config_file["fully_connected_layer"][1]), 
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        y = self.conv(x)
        return self.fc(torch.flatten(y, start_dim=1))


if __name__ == "__main__":
    x = torch.randn((4, 1, 28, 28))

    with open("config.yaml", "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    lenet = LeNet(data)

    y = lenet(x)
    print(y.dtype)