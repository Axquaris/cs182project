from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

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



def sample_img(path_root, img_size, batch_size):
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
        return next(loader)[0]

    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    def create_ds():
        ds = datasets.ImageFolder(path_root, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1)
        return loader
    
    return sample_fn