import torch, os, sys
import numpy as np
from torch import nn
from torchvision import transforms, models
from data_helpers import DATA_DIR, sample_batch

data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
])

val_sampler = sample_batch(os.path.join(DATA_DIR, "val"), None, batch_size=1024, transform=data_transforms)
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

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please specify path to model checkpoint"
    chkpt_path = sys.argv[1]
    checkpoint = torch.load(chkpt_path, map_location=torch.device('cpu'))
    model = models.resnet50()
    model.fc = torch.nn.Linear(2048, 10000)
    model.load_state_dict(checkpoint["net"])
    validate(model)