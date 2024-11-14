import os 
import sys
import matplotlib.pyplot as plt

# Add necessary directories to the path.
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_directory, 'out'))
sys.path.append(os.path.join(parent_directory, 'logs'))
sys.path.append(os.path.join(parent_directory, 'src'))

# Import local modules.
from states import load_solver_package_from_path


def plot_loss_from_solver_path(solver_path):

    # Load solver package and relevant data.
    solver_package = load_solver_package_from_path(solver_path)
    trn_loss = solver_package['trn_loss_history']
    val_loss = solver_package['val_loss_history']
    num_epochs = range(solver_package['running_epoch'])

    # Plot the losses.
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(num_epochs, trn_loss, label='Train Loss')
    ax.plot(num_epochs, val_loss, label='Validation Loss')
    
    ax.grid()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    fig.tight_layout()
    plt.show()

def plot_loss_from_log_file(log_path):
    """ Open a training log file and plot the loss history.

    Args:
        log_path (str): Path to the training log file.

    Returns:
        log_path (str): Path to the training log file.
    """

    # Initialize loss history arrays.
    train_loss = []
    valid_loss = []

    # Read the log file.
    with open(log_path, 'r') as log_file:

        lines = log_file.readlines()

        # Fill in the loss history arrays.
        for line in lines:
            if line.startswith('Train'):
                #train_loss.append(float(line.split('Loss = ')[1]))
                train_loss.append(float(line.split('Loss = ')[1]) / 91)
            elif line.startswith('Valid'):
                #valid_loss.append(float(line.split('Loss = ')[1]))
                valid_loss.append(float(line.split('Loss = ')[1]) / 30)

        num_epochs = range(len(train_loss))

    # Plot the losses.
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(num_epochs, train_loss, label='Train Loss')
    ax.plot(num_epochs, valid_loss, label='Validation Loss')
    
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss History")
    ax.legend()

    fig.tight_layout()
    plt.show()

    return log_path

if __name__ == '__main__':

    log_path = r"logs\train.log"
    plot_loss_from_log_file(log_path)
