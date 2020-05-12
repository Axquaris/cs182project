import os, sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "tiny-imagenet-200")

from data_helpers.dataset import *