
import torch
from torch import nn


class TorchModelTemplate(nn.Module):
    """ A template class that forms the framework for future touch models. """
    def __init__(self, **kwargs):
        """ The parameters of each layer in the touch model are expected. """
        pass

    def forward(self, source, targets, criterion=None):
        """ Returns the output of running data through the model.
        If the criterion function is given, the logits and loss
        of the model is returned, else, only the logits are returned.

        Args:
            source, targets - The independent and dependent variables respectively.
            criterion - the criterion function for calculating loss.

        Returns
        """
        logits = None  # TODO: Implement data passing through the layers.
        if criterion is not None:
            loss = criterion(logits, targets)

            return logits, loss
        else:
            return logits

    def predict(self, data, is_logits=False):
        """ The batch of independent variables for which to make the predictions on.

        Args:
            data - The independent variables. If is_logits == True, then the data is expected to be the logits.
            is_logits - bool. Default False. If True, data is expected to be logits. This is the
                        case during training where the logits and loss comes from .forward, so
                        the logit can be fed directly into .predict instead of for the data
                        to go through .forward again.
        """
        softmax = lambda x: x   # TODO: Implement the softmax function proper.

        # Make predictions.
        if is_logits:
            logits = self.forward(data)
        else:
            logits = data
        probs = softmax(logits)
        pred = torch.argmax(probs, axis=-1).item()

        return pred, prob
