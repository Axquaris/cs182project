import unittest, os
from torchvision.datasets import ImageFolder
import torch
from dataset import sample_batch, gen_base_transform
from config import DATA_DIR


data_transforms = gen_base_transform()
val_sampler = sample_batch(os.path.join(DATA_DIR, "val"), None, batch_size=1, transform=data_transforms)

sample, _ = val_sampler()
print(torch.max(sample))



