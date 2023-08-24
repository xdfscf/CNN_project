import numpy as np


def mse_loss(predictions, targets):

    error = predictions - targets
    squared_error = np.square(error)
    mean_squared_error = np.mean(squared_error)
    return mean_squared_error