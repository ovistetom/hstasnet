import os
import sys
import torch
from torch.utils.data import DataLoader

# Add necessary directories to the path.
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_directory, 'data'))
sys.path.append(os.path.join(parent_directory, 'output'))
sys.path.append(os.path.join(parent_directory, 'hstasnet'))
sys.path.append(os.path.join(parent_directory, 'logs'))
sys.stdout = open(os.path.join('logs', 'train.log'), 'wt')

import losses
from solver import Solver
from dataset import MUSDB18Dataset
from hstasnet import HSTasNet

def define_args():
    """
    Define the default arguments.

    Returns:
        args (dict): A dictionary containing the default arguments.
    """
    args = {
        # Training parameters.
        'batch_size': 1,
        'num_epochs': 2,
        'learning_rate': 0.001,
        'num_workers': 2,
        'continue_from': None,

        # Model parameters.
        'model_name': 'hstasnet',
        'model_path': os.path.join('output', 'models', 'hstasnet.pt'),
        'sources': ['bass', 'drums', 'other', 'vocals'],
        'model_args': {
            'num_sources': 4,
            'num_channels': 2,
            'time_win_size': 1024,
            'time_win_size': 1024,
            'time_hop_size': 512,
            'time_ftr_size': 1500,
            'spec_win_size': 1024,
            'spec_hop_size': 512,
            'spec_fft_size': 1024,
            'rnn_hidden_size': 200,            
            }
        }

    return args


def define_loaders(args):
    """
    Define DataLoaders for the training, validation, and test sets.

    Args:
        args (dict): A dictionary containing the arguments.

    Returns:
        loaders (dict): A dictionary containing the DataLoaders.
    """
        
    root = os.path.join('data', 'musdb18_preprocessed')
    sources = args['sources']

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


def main(args, train = True):

    # Define loaders.
    loaders = define_loaders(args)

    # Define model.
    model = HSTasNet(**args['model_args'])

    # Define criterion.
    criterion = losses.l1_loss

    # Define optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    # Define scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Define solver.
    solver = Solver(model, criterion, optimizer, scheduler, loaders, args)
    solver = solver.train() if train else solver

    return solver


if __name__ == '__main__':

    print("*** START TRAINING ***\n")

    # Define arguments.
    args = define_args()

    # Train the model.
    solver = main(args, train=True)

    print("\n*** FINISHED TRAINING ***")
