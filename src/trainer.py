from tqdm import trange
import torch
from torch import nn
from accelerate import Accelerate

def train_torch(model, tr_iter, num_epoch, criterion, optimizer, path, experiment_name,
                test_every: int, checkpt_every: int, use_accelerate: bool=False,
                device='cpu', te_iter=None, log_to_mlflow=False):
    """ helper function for training pytorch model

    Args:
        path - str. path to save chkpt.
        experiment_name - for checkpoint purpose.
        test_every - int. test with test data after every test_every epochs.
        checkpt_every - int. Checkout the model after every checkpt_every epochs.
        use_accelerate - bool. Default = False. Whether to use hugging face's accelerate library
                         that abstracts the boilerplate code for distributed training.
                         - https://huggingface.co/blog/accelerate-library
                         - https://github.com/huggingface/accelerate
    """
    if use_accelerate:
        accelerator = Accelerator()
        # device = accelerator.device  # don't even need this.

        model, optim, tr_iter = accelerator.prepare(model, optim, tr_iter)

    # Loop through training data
    for epoch in trange(num_epoch):
        if not use_accelerate:
            model = model.to(device)
        model.train()
        model.zero_grad()
        optim = optimizer(model.parameters())

        for i, (source, targets) in tqdm(enumerate(tr_iter)):
            if not use_accelerate:
                source = source.to(device)
                targets = targets.to(device)

            # Train the model
            logits, loss = model(source, targets, criterion)
            if use_accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()
            optim.step()
            # TODO: Everything below
            # Log to mlflow
            # log model config
            # log training config
            # log optim config
            # log metric

        if (epoch + 1) % test_every == 0:
            model.eval()
            if use_accelerate:
                model, _, te_iter = accelerator.prepare(model, optim, te_iter)
            # Loop through test data
            for i, data in tqdm(enumerate(te_iter)):
                if not use_accelerate:
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
