import torch


def load_optimizer(optimizer):
    try:
        if optimizer == "AnyPrecisionAdamW":
            from libs.anyprecision_optimizer import AnyPrecisionAdamW

            return AnyPrecisionAdamW
        elif optimizer.endswith("8bit"):
            import bitsandbytes

            return getattr(bitsandbytes.optim, optimizer)
        else:
            return getattr(torch.optim, optimizer)
    except:
        raise Exception(f"Optimizer {optimizer} does not exist")


def replace_module(model, name, new_module):
    name_parts = name.split(".")
    sub_model = model
    for part in name_parts[:-1]:
        sub_model = getattr(sub_model, part)
    setattr(sub_model, name_parts[-1], new_module)
