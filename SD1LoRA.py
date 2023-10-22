import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import lightning as L
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.fabric.wrappers import _unwrap_objects
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline


from peft import LoraConfig
from peft.utils import get_peft_model_state_dict


def get_training_parameters(config, unet, text_encoder):
    parameters = []
    if config.unet.train:
        unet_lora_config = LoraConfig(**config.unet.peft.parameters)
        unet.add_adapter(unet_lora_config)
        parameters += [
            {
                "params": [p for _, p in unet.named_parameters() if p.requires_grad],
                "lr": config.unet.peft.lr,
            },
        ]
    if config.text_encoder.train:
        te_lora_config = LoraConfig(**config.text_encoder.peft.parameters)
        text_encoder.add_adapter(te_lora_config)
        parameters += [
            {
                "params": [
                    p for _, p in text_encoder.named_parameters() if p.requires_grad
                ],
                "lr": config.text_encoder.peft.lr,
            },
        ]
    return parameters, unet, text_encoder


def save_lora_checkpoint(config, fabric, unet, text_encoder, current_iter=None):
    unet_lora_state_dict = None
    te_lora_state_dict = None
    if config.unet.train:
        unet_lora_state_dict = get_peft_model_state_dict(_unwrap_objects(unet))
    if config.text_encoder.train:
        te_lora_state_dict = get_peft_model_state_dict(_unwrap_objects(text_encoder))

    if current_iter:
        save_file_name = (
            f"{config.name}_{current_iter}_{config.logging.save.every}.safetensors"
        )
    else:
        save_file_name = f"{config.name}.safetensors"

    os.makedirs(f"runs/{config.run_name}/output", exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=f"runs/{config.run_name}/output/",
        weight_name=save_file_name,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=te_lora_state_dict,
        safe_serialization=True,
        is_main_process=fabric.is_global_zero,
    )
