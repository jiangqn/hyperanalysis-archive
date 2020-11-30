import torch
import numpy as np
import random

def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.
    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)