from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from functools import partial
from torchvision import transforms
# https://github.com/google-research/augmix
from data_helpers.augmentations import augmix
from data_helpers.config import DATA_DIR
import os

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def gen_base_transform(im_height=64, im_width=64):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def gen_augmix_transforms(jsd=False, **augmix_kwargs):
    assert not jsd, 'Not implemented yet'
    t = []

    preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    augment = transforms.Lambda(partial(augmix, preprocess, **augmix_kwargs))
    t.append(augment)

    # if train:
        # t.append(transforms.RandomResizedCrop(64))

    # else:
    #     t.append(transforms.CenterCrop(64))

    return transforms.Compose(t)

def sample_batch(path_root, img_size, batch_size, transform=None):
    """
    path_root (string)      the root file path to search for images (.png, .jpg, .mp4). Will recursively
                                search subdirectories
    img_size (int)          Number of pixels to crop the image to (both height and width)
    batch_size (int)        Number of samples to return (all in one batched tensor). -1 returns the whole dataset

    Returns a function used to sample tensors of shape (batch_size, img_size, img_size, 3) randomly from 
    the path provided. Useful for ad-hoc testing
    """
    loaded = False
    loader = None
    def sample_fn():
        nonlocal loaded
        nonlocal loader
        if not loaded:
            loader = iter(create_ds())
            loaded = True
        try:
            return next(loader)
        except StopIteration:
            loader = iter(create_ds())
            return next(loader)

    if not transform:
        transform = gen_base_transform(img_size, img_size)
    
    def create_ds():
        ds = datasets.ImageFolder(path_root, transform=transform)
        bs = batch_size if batch_size > 0 else len(ds)
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        return loader
    
    return sample_fn

def gen_label_to_words_dict():
    words_path = os.path.join(DATA_DIR, "words.txt")
    label_to_words = {}
    with open(words_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            label, words = line.split('\t')[0].strip(), " ".join(line.split('\t')[1:]).strip()
            label_to_words[label] = words

    return label_to_words