"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import glob
import os
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn
from datetime import datetime
from models import *
from data_helpers import DATA_DIR, sample_batch, gen_base_transform, gen_augmix_transforms

# Map string names to load paths for all possible existing models
name_to_model_cls = {
    "baseline" : BaselineResNet,
    "net" : Net
}



def main(args):
    model_name = args.model

    # Once we add more models, update this set here and the input statement
    assert model_name in name_to_model_cls, "Invalid model choice"

    model_cls = name_to_model_cls[model_name]


    # Create a pytorch dataset
    image_count = len(list(glob.glob(os.path.join(DATA_DIR, '**/*.JPEG'), recursive=True)))
    if args.verbose:
        print('Discovered {} images'.format(image_count))
    if args.augment:
        if args.verbose:
            print("Augmenting training data with severity {}".format(args.augment))
        data_transforms = gen_augmix_transforms(severity=int(args.augment))
    else:
        data_transforms = gen_base_transform(args.im_height, args.im_width)
    train_set = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    val_sampler = sample_batch(os.path.join(DATA_DIR, "val"), None, batch_size=32, transform=data_transforms)

    # Create a simple model
    if args.verbose:
        print("loading model...")
    model = model_cls(im_height=args.im_height, im_width=args.im_width, dropout=args.dropout, num_frozen_layers=args.num_frozen_layers).to(device=args.device)
    if args.verbose:
        print("Successfully loaded model")
    # source: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    params_to_update = []
    for _,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optim = torch.optim.Adam(params_to_update, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # data structures to store important info
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    for i in range(args.epochs):
        if args.verbose:
            print("Beginning epoch {0}".format(i))
        
        train_total, train_correct = 0, 0
        for idx, (inputs, targets) in enumerate(train_loader):
            ## Train step 
            inputs = inputs.to(device=args.device)
            targets = targets.to(device=args.device)
            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            training_losses.append(loss.item())

            # TODO: waaaay to verbose bruh
            if args.verbose and idx % 1000 == 0:
                print(f'training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}')
                print("Loss:\t{0}".format(loss.item()))
            
        
        
        ## Validation of model 
        model.eval()
        inputs, targets = val_sampler()
        with torch.no_grad():
            inputs = inputs.to(device=args.device)
            targets = targets.to(device=args.device)
            out = model(inputs)
            loss = criterion(out, targets).item()
            _, pred = out.max(1)
        correct = pred.eq(targets).sum().item()
        total = targets.size(0)
        val_accuracy = correct / total
        model.train()

        ## Logs
        print("Validation at end of epoch {0}".format(i))
        print("Loss:\t{0:.5f}Accuracy:\t{1}".format(loss, val_accuracy))
        validation_accuracies.append(val_accuracy)
        validation_losses.append(loss)
        training_accuracies.append(train_correct / train_total)

        ## Save model
        if args.verbose:
            print("Saving model checkpoint at end of epoch {0}".format(i))
        torch.save({
            'net': model.state_dict(),
            't_loss' : training_losses,
            't_acc' : training_accuracies,
            'v_loss' : validation_losses,
            'v_acc' : validation_accuracies
        }, os.path.join(args.output_dir, 'epoch_{0}_checkpoint.pt'.format(i)))

       


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--model', default="baseline")
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-a', '--augment', default=1, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('-h', '--im_height', default=64, type=int)
    parser.add_argument('-w', '--im_width', default=64, type=int)
    parser.add_argument('-d', '--dropout', default=0.6, type=float)
    parser.add_argument('-n', '--num-frozen-layers', default=6, type=int)
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()

    args.device = None
    if args.gpu and torch.cuda.is_available():
        if args.verbose:
            print("Running on GPU")
        args.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        if args.verbose:
            print("Running on CPU")
        args.device = torch.device('cpu')

    # Where intermediate checkpoints will be stored
    timestamp = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    args.output_dir = os.path.join(OUTPUT_DIR, "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(args.model, args.num_frozen_layers, args.dropout, args.epochs, args.batch_size, args.learning_rate, timestamp))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)



    main(args)
