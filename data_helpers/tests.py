import unittest, os
from torchvision.datasets import ImageFolder
from dataset import sample_batch, gen_base_transform
from config import DATA_DIR


data_transforms = gen_base_transform()
val_sampler = sample_batch(os.path.join(DATA_DIR, "val"), None, batch_size=6000, transform=data_transforms)

sample_1, targets_1 = val_sampler()
sample_2, targets_2 = val_sampler()
sample_3, _ = val_sampler()
sample_4, _ = val_sampler()
sample_5, _ = val_sampler()

print(sample_1.shape)
print(sample_2.shape)
print(sample_3.shape)
print(sample_4.shape)
print(sample_5.shape)



