import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchvision
from config import BASELINE_PATH as SAVE_PATH

# How many layers (out of 10) will NOT be updated
NUM_FROZEN_LAYERS = 6
# How many logits (classes) to have in the last layer
NUM_LOGITS = 200
# The input number to the last layer (found by printing the model)
NUM_INPUT = 2048



if __name__ == '__main__':
    base_model = torchvision.models.resnet50(pretrained=True)

    # Freezing layers
    child_num = 1
    for child in base_model.children():
        if child_num < NUM_FROZEN_LAYERS:
            for param in child.parameters():
                param.requires_grad = False
        child_num += 1

    # Updating number of logits in last layer
    base_model.fc = torch.nn.Linear(NUM_INPUT, NUM_LOGITS)

    # Writing model to disc
    torch.save(base_model, SAVE_PATH)
