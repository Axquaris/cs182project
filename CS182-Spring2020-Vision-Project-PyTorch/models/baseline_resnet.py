import torch
import torchvision

base_model = torchvision.models.resnet50(pretrained=True)

for name, child in base_model.named_children():
    for name2, params in child.named_parameters():
        print(name, name2)