import os, sys

from data_helpers.dataset import *

def get_data_dir(data_dir):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", data_dir)
