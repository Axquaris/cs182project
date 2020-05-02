"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import pathlib
import os
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn

parser = argparse.ArgumentParser()

parser.add_argument('--model', default="baseline")
args = parser.parse_args()

PATH = args.model

# Once we add more models, update this set here and the input statement
assert PATH in {"baseline"}, "Invalid model choice"

if PATH == "baseline":
    PATH = "baseline_model_modified"
PATH = os.path.join("models", PATH)

def main():
    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = 32
    im_height = 64
    im_width = 64
    num_epochs = 1

    data_transforms = transforms.Compose([
        transforms.Resize((im_height, im_width)),
        transforms.CenterCrop((im_height, im_width)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    # Create a simple model
    model = torch.load(PATH)
    # source: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    params_to_update = []
    for _,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optim = torch.optim.Adam(params_to_update)
    criterion = nn.CrossEntropyLoss()
    for i in range(num_epochs):
        train_total, train_correct = 0,0
        for idx, (inputs, targets) in enumerate(train_loader):
            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            print("\r", end='')
            print(f'training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}', end='')
        torch.save({
            'net': model.state_dict(),
        }, 'epoch_{0}_checkpoint.pt'.format(i))


if __name__ == '__main__':
    main()
