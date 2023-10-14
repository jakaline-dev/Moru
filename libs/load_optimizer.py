import torch


def load_optimizer(optimizer):
    try:
        if optimizer.endswith("8bit"):
            import bitsandbytes

            return getattr(bitsandbytes.optim, optimizer)
        else:
            return getattr(torch.optim, optimizer)
    except:
        raise Exception(f"Optimizer {optimizer} does not exist")
