from tqdm import trange
import torch
from torch import nn

def train_torch(model, tr_iter, num_epoch, criterion, optimizer, path, experiment_name,
                test_every: int, checkpt_every: int,
                device='cpu', te_iter=None, log_to_mlflow=False):
    """ helper function for training pytorch model

    Args:
        path - str. path to save chkpt.
        experiment_name - for checkpoint purpose.
        test_every - int. test with test data after every test_every epochs.
        checkpt_every - int. Checkout the model after every checkpt_every epochs.
    """
    # Loop through training data
    for epoch in trange(num_epoch):
        model = model.to(device)
        model.train()
        model.zero_grad()

        for i, data in tqdm(enumerate(tr_iter)):
            data = data.to(device)

            # Train the model
            logits, loss = model(data, criterion)
            optimizer.step()
            # TODO: Everything below
            # Log to mlflow
            # log model config
            # log training config
            # log optimizer config
            # log metric

        if (epoch + 1) % test_every == 0:
            model.eval()
            # Loop through test data
            for i, data in tqdm(enumerate(te_iter)):
                data = data.to(device)
                logits, loss = model(data, criterion)
                pred, prob = model.predict(logits, is_logits=True)
                # TODO: Refine this.


        # Checkpoint every
        if (epoch + 1) % checkpt_every == 0:
            chkpt_name = f'{path}/{experiment_name}/epoch{epoch}.pt'
            # TODO Everything below.
            # Assign model name and directory
            # Checkpoint model.state_dict()
            # Checkpoint training config
            # Checkpoint optiimizer config
            # Checkpoint number of epoch
