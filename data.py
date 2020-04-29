from functools import partial

from torchvision import transforms

# https://github.com/google-research/augmix
from augmentations import augmix


def generate_transforms(train=True, jsd=False, **augmix_kwargs):
    assert not jsd, 'Not implemented yet'
    t = []

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    augment = transforms.Lambda(partial(augmix, preprocess, **augmix_kwargs))
    t.append(augment)

    if train:
        t.append(transforms.RandomResizedCrop(224))
        t.append(transforms.RandomHorizontalFlip())
    else:
        t.append(transforms.CenterCrop(224))

    return transforms.Compose(t)


if __name__ == '__main__':
    print(generate_transforms())
