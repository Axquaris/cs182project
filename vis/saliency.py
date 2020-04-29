import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import requests
from PIL import Image

class SaliencyMapper:

    def __init__(self, module, images):
        self.module = module
        self.images = images
        self.saliency = {}

    def preprocess(image, size=224):
        transform = T.Compose([
            T.Resize((size,size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x[None]),
        ])
        return transform(image)

    def deprocess(image):
        transform = T.Compose([
            T.Lambda(lambda x: x[0]),
            T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            T.ToPILImage(),
        ])
        return transform(image)

    def show_img(PIL_IMG):
        plt.imshow(np.asarray(PIL_IMG))

    def process():
        for img in self.images:
            i = preprocess(img)
            module.eval()
            i.requires_grad_()
            scores = model(i)

            score_max_index = scores.argmax()
            score_max = scores[0,score_max_index]

            score_max.backward()

            saliency, _ = torch.max(X.grad.data.abs(),dim=1)

            self.saliency[img] = saliency

            # code to plot the saliency map as a heatmap
            plt.imshow(saliency[0], cmap=plt.cm.hot)
            plt.axis('off')
            plt.show()

    def get_saliency():
        return self.saliency
