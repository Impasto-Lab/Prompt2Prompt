"""Pick the compute device and dtype from command-line names."""
import torch


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name):
    # float32 by default: the 32-layer text encoder is not tested in half precision.
    return torch.float32 if name == "auto" else getattr(torch, name)
