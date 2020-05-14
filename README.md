This is some sample code for the CS 182/282 Computer Vision project (Tensorflow 2). It has the following files:


README.txt - This file
requirements.txt - The python requirments necessary to run this project

train_sample.py - A sample training file which trains a simple model on the data, and save the checkpoint to be loaded
                  in the test_submission.py file.

test_submission.py - A sample file which will return an output for every input in the eval.csv

eval.csv - An example test file

data/get_data.sh - A script which will download the tiny-imagenet data into the data/tiny-imagenet-200 file


## Project Evaluation
* Download the final model from the google drive [here](https://drive.google.com/open?id=1DYb9boOv-XIT-rv7fv-hmYeE2hW8R46c)
  * The model in question should be under `final_model/latest.pt`
* Copy the file into the root project directy (i.e. `PROJ_ROOT/latest.pt`)
* Create a virtual environment
* `pip install -r requirements.txt`
* `python test_submission_torch.py [path/to/csv]`

Note: You should be using Python 3 to run this code.
