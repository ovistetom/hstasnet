import matplotlib.pyplot as plt
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

if __name__ == '__main__':

    # solver_path = r"..\output\solvers\hstasnet01.pkl"
    # plot_loss_from_solver_path(solver_path)

    pass
