import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(im_height * im_width * 3, 128)
        self.layer2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class BaselineResNet(nn.Module):
    def __init__(self, num_classes=200, im_height=64, im_width=64, num_frozen_layers=6, dropout=0.6):
        super(BaselineResNet, self).__init__()
        assert num_frozen_layers >= 0 and num_frozen_layers <= 10, "Number of frozen layers must be between 0 and 10"
        self.base_model = torchvision.models.resnet50(pretrained=True)

        # Freezing layers
        child_num = 0
        for child in self.base_model.children():
            if child_num < num_frozen_layers:
                for param in child.parameters():
                    param.requires_grad = False
            child_num += 1

        self.dropout = nn.Dropout(p=dropout)
        self.final_fc = nn.Linear(self.base_model.fc.out_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.final_fc(x)
        return x

class Generator(nn.Module):
    def __init__(self, im_height=64, im_width=64, bottleneck_length=2):
        super(Generator, self).__init__()
        input_length = im_height * im_width * 3
        self.downsample_1 = nn.Linear(input_length, input_length//2)
        self.downsample_2 = nn.Linear(input_length//2, input_length//4)
        self.upsample_1 = nn.Linear(input_length//4, input_length//2)
        self.upsample_2 = nn.Linear(input_length//2, input_length)

        self.downsampling_layers = [self.downsample_1, self.downsample_2]
        self.upsampling_layers = [self.upsample_1, self.upsample_2]

        self.final = nn.Sigmoid()

    def forward(self, x):
        shape = x.shape
        x = x.flatten(1)
        for layer in self.downsampling_layers:
            x = layer(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        x = self.final(x)
        x = x.view(shape)
        return x

class Generator(nn.Module):
    def __init__(self, in_height=64, in_width=64, bottleneck_length=3):
        super(Generator, self).__init__()
        downsampling_layers = [
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten()
        ]
        self.downsampling = nn.Sequential(downsampling_layers)
        self.fc = nn.Sequential(nn.Linear((in_width*in_height)//16, (in_width*in_height)//16), nn.ReLU(True))
        upsampling_layers = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d((in_width*in_height)//8, 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. 32 x 8 x 8
            nn.ConvTranspose2d( 32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # state size. 16 x 16 x 16
            nn.ConvTranspose2d( 16, 8, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # state size. 8 x 32 x 32
            nn.ConvTranspose2d( 8, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        ]
        self.upsampling = nn.Sequential(upsampling_layers)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.fc(x)
        x = self.upsampling(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, im_height=64, im_weidth=64):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
                    nn.Linear(8192, 1),
                    nn.Sigmoid())

    def forward(self, x):
         x = self.layer1(x)
         x = self.layer2(x)
         x = x.reshape(x.size(0), -1)
         x = self.fc(x)
         return x
