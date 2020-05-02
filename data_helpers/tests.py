import unittest, os
from torchvision.datasets import ImageFolder
from dataset import TransformsDataset

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
        self.base_ds = ImageFolder(self.data_dir, None)

    def test_transform_dataset(self):
        ## TODO
        pass


