import glob, os, argparse
import numpy as np
import torch, torchvision
import torchvision.transforms as transforms

from torch import nn
from datetime import datetime
from models import *
from data_helpers import get_data_dir, sample_batch, gen_base_transform

def main(args):
    DATA_DIR = get_data_dir(args.data_dir)
    data_transforms = gen_base_transform(args.im_height, args.im_width)
    train_set = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, "train"), data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    bottleneck_length = args.bottleneck_length
    if args.verbose:
        print("loading models ...")
    dis = Discriminator().to(device=args.device)
    gen = Generator(bottleneck_length=bottleneck_length).to(device=args.device)
    if args.verbose:
        print("Models loaded successfully")

    optimG = torch.optim.Adam(gen.parameters(), lr=args.gen_lr)
    optimD = torch.optim.Adam(dis.parameters(), lr=args.dis_lr)
    criterion = nn.BCELoss()

    real_label = 1
    fake_label = 0

    gen_losses = []
    dis_losses = []
    iters = 0

    for epoch in range(args.epochs):
        if args.verbose:
            print(f"Beginning epoch {epoch}")

        for idx, (inputs, _) in enumerate(train_loader):
            optimD.zero_grad()
            real_inputs = inputs.to(device=args.device)
            real_size = real_inputs.size(0)

            targets = torch.full((real_size,), real_label, device=args.device)
            outputs = dis(real_inputs).view(-1)

            loss_real = criterion(outputs, targets)
            loss_real.backward()

            D_x = outputs.mean().item()

            noise = torch.randn_like(real_inputs, device=args.device)
            fake = torch.add(gen(noise), real_inputs)
            targets.fill_(fake_label)
            outputs = dis(fake.detach()).view(-1)

            loss_fake = criterion(outputs, targets)
            loss_fake.backward()
            D_G_z1 = outputs.mean().item()

            loss = loss_real + loss_fake

            optimD.step()

            optimG.zero_grad()
            targets.fill_(real_label)
            outputs = dis(fake).view(-1)

            loss_gen = criterion(outputs, targets)
            loss_gen.backward()
            optimG.step()

            print(loss_gen)
            print(loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-gen_lr', default=1e-4, type=float)
    parser.add_argument('-dis_lr', default=1e-4, type=float)
    parser.add_argument('-h', '--im_height', default=64, type=int)
    parser.add_argument('-w', '--im_width', default=64, type=int)
    parser.add_argument('--bottleneck_length', default=2, type=int)
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--data_dir', default="tiny-imagenet-200", type=str)
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
    args.output_dir = os.path.join(OUTPUT_DIR, "GAN_{0}_{1}_{2}_{3}_{4}".format(args.epochs, args.batch_size, args.gen_lr, args.dis_lr, timestamp))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    main(args)
