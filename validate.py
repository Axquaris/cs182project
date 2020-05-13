import torch, os, sys, torchvision
import argparse
import numpy as np
from torch import nn
from torchvision import transforms, models, datasets
from data_helpers import DATA_DIR, sample_batch, gen_base_transform
from models import BaselineResNet, Generator

data_transforms_legacy = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
])

data_transforms = gen_base_transform()

dummy_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=data_transforms)
num_classes = len(dummy_ds.classes)

del dummy_ds
val_sampler = sample_batch(os.path.join(DATA_DIR, "val"), None, batch_size=1024, transform=data_transforms)
train_sampler = sample_batch(os.path.join(DATA_DIR, "train"), None, batch_size=5, transform=data_transforms)

criterion = nn.CrossEntropyLoss()

# Validation of model
def validate(model):
    model.eval()
    inputs, targets = val_sampler()
    with torch.no_grad():
        out = model(inputs)
        loss = criterion(out, targets).item()
        _, pred = out.max(1)
    correct = pred.eq(targets).sum().item()
    total = targets.size(0)
    val_accuracy = correct / total

    print("Loss:\t{0}Accuracy:\t{1}".format(loss, val_accuracy))

def validate_gan(gen, clf):
    model.eval()
    inputs, true_labels = train_sampler()
    with torch.no_grad():
        pertub = gen(inputs)
        x_adv = torch.add(pertub, inputs)
        target_labels = torch.fmod(true_labels + 1, num_classes)
        predicted_labels_adv = torch.argmax(clf(x_adv), dim=1)
        predicted_labels = torch.argmax(clf(inputs), dim=1)
        print("input norm",torch.norm(inputs))
        print("pertubation norm", torch.norm(pertub))
        print("True labels", true_labels)
        print("Target labels", target_labels)
        print("Predicted labels on input", predicted_labels)
        print("Predicted labels on corrupted", predicted_labels_adv)

        torchvision.utils.save_image(torch.cat([inputs, x_adv], dim=0), fp="./gan_results.png", nrow=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', type=str)
    parser.add_argument('-g', '--gan-path', type=str)
    parser.add_argument('--gan', action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model = BaselineResNet()
    model.load_state_dict(checkpoint["net"])
    if args.gan:
        assert args.gan_path, "must specify path to gan checkpoint to evaluate gan"
        gan_checkpoint = torch.load(args.gan_path, map_location=torch.device('cpu'))
        gen = Generator()
        gen.load_state_dict(gan_checkpoint['gen'])
        validate_gan(gen, model)
    else:
        validate(model)