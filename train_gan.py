import glob, os, argparse
import numpy as np
import torch, torchvision
import torchvision.transforms as transforms

from torch import nn
from datetime import datetime
from models import *
from data_helpers import DATA_DIR, sample_batch, gen_base_transform

def main(args):
    data_transforms = gen_base_transform(args.im_height, args.im_width)
    train_set = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, "train"), data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    num_classes = len(train_set.classes)

    bottleneck_length = args.bottleneck_length
    if args.verbose:
        print("loading models ...")
    dis = Discriminator().to(device=args.device)
    gen = Generator(bottleneck_length=bottleneck_length).to(device=args.device)

    # Load pre-trained classifier
    checkpoint = torch.load(args.classifier_path, map_location=args.device)
    classifier = BaselineResNet()
    classifier.load_state_dict(checkpoint["net"])
    classifier = classifier.to(device=args.device)
    if args.verbose:
        print("Models loaded successfully")

    optimG = torch.optim.Adam(gen.parameters(), lr=args.gen_lr)
    optimD = torch.optim.Adam(dis.parameters(), lr=args.dis_lr)
    criterion = nn.BCELoss()
    whitebox_criterion = nn.CrossEntropyLoss()

    real_label = 1
    fake_label = 0

    gen_losses = []
    dis_losses = []
    iters = 0

    for epoch in range(args.epochs):
        if args.verbose:
            print(f"Beginning epoch {epoch}")

        for idx, (inputs, real_targets) in enumerate(train_loader):
            optimD.zero_grad()
            real_inputs = inputs.to(device=args.device)
            real_size = real_inputs.size(0)

            dis_targets = torch.full((real_size,), real_label, device=args.device)
            dis_outputs = dis(real_inputs).view(-1)

            loss_real = criterion(dis_outputs, dis_targets)
            loss_real.backward()


            gen_out = gen(real_inputs)
            fake = torch.add(gen_out, real_inputs)
            dis_targets.fill_(fake_label)
            outputs = dis(fake.detach()).view(-1)

            loss_fake = criterion(outputs, dis_targets)
            loss_fake.backward()
            D_G_z1 = outputs.mean().item()

            loss = loss_real + loss_fake
            dis_losses.append(loss.item())

            optimD.step()

            optimG.zero_grad()
            classifier.zero_grad()
            dis_targets.fill_(real_label)
            outputs = dis(fake).view(-1)

            # Target labels for our whitebox attack (just add one for simplicity)
            adv_targets = torch.fmod(real_targets + 1, num_classes)

            loss_gen_gan = criterion(outputs, dis_targets)
            loss_gen_adv = whitebox_criterion(classifier(fake), adv_targets)

            pertubation_norms = torch.norm(gen_out.view(args.batch_size, -1), p=2, dim=1)
            loss_gen_hinge = torch.mean(torch.max(pertubation_norms - args.c, torch.zeros_like(pertubation_norms)))
            loss_gen = loss_gen_adv + args.alpha * loss_gen_gan + args.beta * loss_gen_hinge
            loss_gen.backward()
            optimG.step()

            loss_gen_info = {
                "hinge_loss" : loss_gen_hinge.item(),
                "adversarial_loss" : loss_gen_adv.item(),
                "fooling_loss" : loss_gen_gan.item()
            }

            if args.verbose and iters % args.print_every == 0:
                print("Generator loss", loss_gen_info)
                print("Discriminator loss", loss.item())

            gen_losses.append(loss_gen_info)

            iters += 1


        ## Save model
        if args.verbose:
            print("Saving model checkpoint at end of epoch {0}".format(epoch))
        torch.save({
            'gen': gen.state_dict(),
            'dis' : dis.state_dict(),
            'dis_loss' : dis_losses,
            'gen_loss' : gen_losses,
        }, os.path.join(args.output_dir, 'epoch_{0}_checkpoint.pt'.format(epoch)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-p', '--print-every', default=100, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-gen_lr', default=1e-4, type=float)
    parser.add_argument('-dis_lr', default=1e-4, type=float)
    parser.add_argument('-h', '--im_height', default=64, type=int)
    parser.add_argument('-w', '--im_width', default=64, type=int)
    parser.add_argument('-clf', '--classifier-path', required=True, type=str)
    parser.add_argument('--bottleneck_length', default=2, type=int)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--c', default=1, type=float)
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
