from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class TransformsDataset(Dataset):

    def __init__(self, ds, transforms):
        self.base_ds = ds
        self.transforms = transforms

    def __getitem__(self, idx):
        ds_idx, trans_idx = idx / len(self.transforms), idx % len(self.transforms)
        base_img, label = self.base_ds[ds_idx]
        return (self.transforms[trans_idx](base_img), label)

    def __len__(self):
        return len(self.base_ds) * len(transforms)

def gen_base_transform(im_height=64, im_width=64):
    return transforms.Compose([
        transforms.Resize((im_height, im_width)),
        transforms.CenterCrop((im_height, im_width)),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])

def sample_batch(path_root, img_size, batch_size, transform=None):
    """
    path_root (string)      the root file path to search for images (.png, .jpg, .mp4). Will recursively
                                search subdirectories
    img_size (int)          Number of pixels to crop the image to (both height and width)
    batch_size (int)        Number of samples to return (all in one batched tensor)

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
        return next(loader)

    if not transform:
        transform = gen_base_transform(img_size, img_size)
    
    def create_ds():
        ds = datasets.ImageFolder(path_root, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return loader
    
    return sample_fn