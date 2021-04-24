from tqdm import trange
import torch
from torch import nn

def train_torch(model, tr_iter, num_epoch, criterion, optimizer, path, experiment_name, device='cpu', te_iter=None, log_to_mlflow=False):
    """ helper function for training pytorch model 
    
    Args:
        path - str. path to save chkpt.
        experiment_name - for checkpoint purpose.
    """
    # Loop through training data
    for epoch in trange(num_epoch):
        model = model.to(device)
        model.train()
        model.zero_grad()

        for i, data in tqdm(enumerate(tr_iter)):
            # Train the model
            output = model(data)
            loss = criterion()
            optimizer.step()
            # Log to mlflow
            # log model config
            # log training config
            # log optimizer config
            # log metric

        # Loop through test data
        for i, data in tqdm(enumerate(te_iter)):
            model.eval()

        # Checkpoint
        chkpt_name = f'{path}/{experiment_name}/epoch{epoch}.pt'
        # Assign model name and directory
        # Checkpoint model.state_dict()
        # Checkpoint training config
        # Checkpoint optiimizer config
        # Checkpoint number of epoch
