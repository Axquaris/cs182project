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
