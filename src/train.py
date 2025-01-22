import os
import sys
import torch
from torch.utils.data import DataLoader
from datetime import datetime


# Add necessary directories to the path.
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_directory, 'data'))
sys.path.append(os.path.join(parent_directory, 'out'))
sys.path.append(os.path.join(parent_directory, 'hstasnet'))
sys.path.append(os.path.join(parent_directory, 'logs'))
sys.stdout = open(os.path.join('logs', 'train.log'), 'wt')


import losses
from solver import Solver
from dataset import MUSDB18Dataset
from hstasnet import HSTasNet


def define_args():
    """Define the training parameters.

    Returns:
        args (dict): A dictionary containing the parameters for the training routine and the model.
    """
    args = {
        # Training parameters.
        'solver_path': os.path.join('out', 'solvers', f"hstasnet_{datetime.today().strftime('%Y%m%d')}.pkl"),
        'continue_from': None, #os.path.join('out', 'models', 'hstasnet_20250115.pt'),
        # TODO: add support for resuming training from a checkpoint.
        'batch_size': 12,
        'num_epochs': 100,
        'num_workers': 4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

        # Optimizer parameters.
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,

        # Model parameters.
        'model_name': 'hstasnet',
        'model_path': os.path.join('out', 'models', f"hstasnet_{datetime.today().strftime('%Y%m%d')}.pt"),
        'model_srcs': ['bass', 'drums', 'other', 'vocals'],
        'model_args': {
            'num_sources': 4,
            'num_channels': 2,
            'time_win_size': 1024,
            'time_hop_size': 512,
            'time_ftr_size': 512,
            'spec_win_size': 1024,
            'spec_hop_size': 512,
            'spec_fft_size': 1024,
            'rnn_hidden_size': 512,            
            },

        # Other parameters.
        'log_path': os.path.join('out', 'logs', f"hstasnet_{datetime.today().strftime('%Y%m%d')}.log"),
        'save_every': 5,
        }

    return args


def define_loaders(args):
    """Define DataLoaders for the training, validation, and test sets.

    Args:
        args (dict): Dictionary containing the training parameters.

    Returns:
        loaders (dict): Dictionary containing the DataLoaders.
    """
        
    root = os.path.join('/home/ovistetom/Documents/Databases_Local/MUSDB18', 'musdb18hq_augmented')
    sources = args['model_srcs']

    trn_dataset = MUSDB18Dataset(root, 'train', sources)
    val_dataset = MUSDB18Dataset(root, 'valid', sources)
    tst_dataset = MUSDB18Dataset(root, 'test', sources)

    # Define DataLoaders.
    trn_loader = DataLoader(trn_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    tst_loader = DataLoader(tst_dataset, batch_size=args['batch_size'], shuffle=False)

    # Store DataLoaders in a dictionary.
    loaders = {
        'trn_loader': trn_loader,
        'val_loader': val_loader,
        'tst_loader': tst_loader,
        }
    
    return loaders


def main(args, train=True):
    """Define model, loaders, optimizer, criterion, scheduler, solver and launch training.

    Args:
        args (dict): Dictionary containing the training parameters.

    Returns:
        solver (Solver): Solver instance containing the training information, e.g. model and loss history.
    """

    # Define loaders.
    loaders = define_loaders(args)

    # Define model.
    model = HSTasNet(**args['model_args'])
    os.makedirs(os.path.dirname(args['model_path']), exist_ok=True)

    # Define criterion.
    criterion = losses.l1_loss

    # Define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    # Define scheduler.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0) 
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.125, patience=3)

    # Define solver.
    solver = Solver(model, criterion, optimizer, scheduler, loaders, args, device=args['device'])
    os.makedirs(os.path.dirname(args['solver_path']), exist_ok=True)

    # Train model.
    solver = solver.train() if train else solver

    # Save log file.
    os.makedirs(os.path.dirname(args['log_path']), exist_ok=True)
    os.rename(src='logs/train.log', dst=args['log_path'])

    return solver


if __name__ == '__main__':

    # Empty the GPU cache.
    torch.cuda.empty_cache()

    print("*** START TRAINING ***\n")

    # Read parameters for the training routine and the model.
    args = define_args()

    # Train the model.
    solver = main(args)

    print("\n*** FINISHED TRAINING ***")
