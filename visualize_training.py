import matplotlib.pyplot as plt
import torch, sys

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please specify path to model checkpoint"
    chkpt_path = sys.argv[1]
    checkpoint = torch.load(chkpt_path, map_location=torch.device('cpu'))
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
    plt.show()

    # print("v_losses", v_losses)
    # print("t_losses", t_losses)
    # print("v_accuracies", v_accuracies)
    # print("t_accuracies", t_accuracies)