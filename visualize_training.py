import matplotlib.pyplot as plt
import torch, sys
import argparse
import numpy as np


def visualize_clf(checkpoint):
    t_losses = checkpoint['t_loss']
    t_accuracies = checkpoint['t_acc']
    v_losses = checkpoint['v_loss']
    v_accuracies = checkpoint['v_acc']

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2)
    ax1.title.set_text("Training loss per Iter")
    ax1.plot(t_losses)
    ax2.title.set_text("Training accuracy per Epoch")
    ax2.plot(t_accuracies)
    ax3.title.set_text("Validation loss per Epoch")
    ax3.plot(v_losses)
    ax4.title.set_text("Validation Accuracy per Epoch")
    ax4.plot(v_accuracies)

    fig.subplots_adjust(hspace=1.0)

def visualize_gan(checkpoint):
    dis_losses = checkpoint['dis_loss']
    gen_losses = checkpoint['gen_loss']

    adv_losses = [gen_loss['adversarial_loss'] for gen_loss in gen_losses]
    hinge_losses = [gen_loss['hinge_loss'] for gen_loss in gen_losses]
    fooling_losses = [gen_loss['fooling_loss'] for gen_loss in gen_losses]

    num_iters = len(adv_losses)

    x = np.linspace(0, num_iters, num_iters)
    plt.plot(x, dis_losses, label="Discriminator Loss")
    plt.plot(x, adv_losses, label="Adversarial Loss")
    plt.plot(x, fooling_losses, label="Fooling Loss")
    plt.plot(x, hinge_losses, label="Pertubation Regularization Loss")

    plt.legend()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan", action="store_true")
    parser.add_argument("-p", "--checkpoint-path", type=str)
    args = parser.parse_args()
    chkpt_path = args.checkpoint_path
    checkpoint = torch.load(chkpt_path, map_location=torch.device('cpu'))

    if args.gan:
        visualize_gan(checkpoint)
    else:
        visualize_clf(checkpoint)
    
    plt.show()