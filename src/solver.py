import torch
from tqdm import tqdm
from states import load_solver_package_from_path


class Solver:
    """
    A class to train and evaluate a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        criterion (nn.Module): The loss function to be optimized.
        optimizer (optim.Optimizer): The optimizer to be used for training.
        scheduler (optim.lr_scheduler._LRScheduler): The learning rate scheduler to be used for training.
        loaders (dict): A dictionary containing the DataLoaders for the training, validation, and test sets.
        args (dict): A dictionary containing additional arguments.
        device (str, optional): The device to use for training. Defaults to 'cpu'.
    """

    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 loaders,
                 args,
                 device='cpu',
                 ):
    
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = args['num_epochs']
        self.loaders = loaders
        self.device = device
        self.model.to(device)
        self.trn_loss_history = torch.zeros(self.num_epochs, device=device)
        self.val_loss_history = torch.zeros(self.num_epochs, device=device)
        self._reset()

    def train(self):

        for epoch in range(self.running_epoch, self.num_epochs):

            # Train.
            self.model.train()
            trn_loss = self._run_one_trn_epoch()

            print(f'Train Summary | Epoch {epoch+1:02d} | Loss = {trn_loss:.3f}')

            # Validate.
            self.model.eval()
            with torch.no_grad():             
                val_loss = self._run_one_val_epoch()

            print(f'Valid Summary | Epoch {epoch+1:02d} | Loss = {val_loss:.3f}')

            # Update scheduler.
            if self.scheduler is not None:
                self.scheduler.step()
                last_lr = self.scheduler.get_last_lr()[0]
                print(f'Learning rate = {last_lr:.6f}')

            # Save model.
            self.trn_loss_history[epoch] = trn_loss
            self.val_loss_history[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_to_path(self.args['model_path'])
                print(f"Best model saved at {self.args['model_path']}")

            self.running_epoch += 1
            print('\n')

        return self

    def _run_one_trn_epoch(self):

        running_loss = 0.0
        for i, batch_i in enumerate(tqdm(self.loaders['trn_loader'], "Training epoch")):

            # Get the inputs and targets.
            batch_mixture, batch_sources = batch_i
            batch_mixture = batch_mixture.to(self.device)
            batch_sources = batch_sources.to(self.device)

            # Forward pass.
            batch_length = batch_sources.size(-1)
            batch_outputs = self.model(batch_mixture, length=batch_length)

            # Compute loss.
            loss = self.criterion(batch_outputs, batch_sources)

            # Backward pass and optimization.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss
    
    def _run_one_val_epoch(self):        

        running_loss = 0.0
        for i, batch_i in enumerate(tqdm(self.loaders['val_loader'], "Validating epoch")):

            # Get the inputs and targets.
            batch_mixture, batch_sources = batch_i
            batch_mixture = batch_mixture.to(self.device)
            batch_sources = batch_sources.to(self.device)

            # Forward pass.
            batch_length = batch_sources.size(-1)
            batch_outputs = self.model(batch_mixture, length=batch_length)

            # Compute loss.
            loss = self.criterion(batch_outputs, batch_sources)

            running_loss += loss.item()

        return running_loss    
    
    def test(self):

        self.model.eval()
        with torch.no_grad():
            tst_loss = self._run_one_tst_epoch()

            print(f'Test Summary | Loss = {tst_loss:.3f}')
            
    def _run_one_tst_epoch(self):

        running_loss = 0.0
        for i, batch_i in enumerate(tqdm(self.tst_loader, "Testing epoch")):

            # Get the inputs and targets.
            batch_mixture, batch_sources = batch_i
            batch_mixture = batch_mixture.to(self.device)
            batch_sources = batch_sources.to(self.device)

            # Forward pass.
            batch_length = batch_sources.size(-1)
            batch_outputs = self.model(batch_mixture, length=batch_length)

            # Compute loss.
            loss = self.criterion(batch_outputs, batch_sources)

            running_loss += loss.item()

        return running_loss        

    def _reset(self):

        if self.args['continue_from']:
            print(f"Loading checkpoint model: {self.args['continue_from']}")     
            package = load_solver_package_from_path(self.args['continue_from'])
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optimizer_dict'])
            self.scheduler.load_state_dict(package['scheduler_dict'])
            self.trn_loss_history[:self.running_epoch] = package['trn_loss_history'][:self.running_epoch]
            self.val_loss_history[:self.running_epoch] = package['val_loss_history'][:self.running_epoch]
            self.running_epoch = package['running_epoch']
        else:
            self.running_epoch = 0

        self.prev_val_loss = float('inf')
        self.best_val_loss = float('inf')

    def _init_args_kwargs(self):

        args = [
            ]

        kwargs = {
            'device': self.device,
            }

        return args, kwargs
    
    def serialize(self):
        """
        Serialize the solver into a dictionary.
        
        Args:    
            solver (Solver): The solver to serialize.

        Returns:
            dict: A dictionary containing the solver's class, arguments, keyword arguments, and state.
        """

        package = {
            'state_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'scheduler_dict': self.scheduler.state_dict(),
            'running_epoch': self.running_epoch,
            'trn_loss_history': self.trn_loss_history,
            'val_loss_history': self.val_loss_history,
            }
        
        return package
    
    def save_to_path(self, path):
        """
        Save the solver to a given file path.

        Args:
            path (str): The file path to save the solver to.
        """
        package = self.serialize()
        torch.save(package, path)
