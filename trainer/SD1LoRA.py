import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import lightning as L
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.fabric.wrappers import _unwrap_objects
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

from libs.convert_LoRA import get_module_kohya_state_dict
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file


def get_training_parameters(config, unet, text_encoder):
    parameters = []
    if config.unet_peft:
        for unet_peft in config.unet_peft:
            unet_lora_config = LoraConfig(**unet_peft.parameters)
            unet.add_adapter(unet_lora_config)
            parameters += [
                {
                    "params": [
                        p for _, p in unet.named_parameters() if p.requires_grad
                    ],
                    "lr": unet_peft.lr,
                },
            ]
    if config.text_encoder_peft:
        for te_peft in config.text_encoder_peft:
            te_lora_config = LoraConfig(**te_peft.parameters)
            text_encoder.add_adapter(te_lora_config)
            parameters += [
                {
                    "params": [
                        p for _, p in text_encoder.named_parameters() if p.requires_grad
                    ],
                    "lr": te_peft.lr,
                },
            ]
    return parameters, unet, text_encoder


def save_lora_checkpoint(config, fabric, unet, text_encoder, current_iter=None):
    if current_iter:
        save_file_name = (
            f"{config.name}_{current_iter}_{config.logging.save.every}.safetensors"
        )
    else:
        save_file_name = f"{config.name}.safetensors"

    unet_lora_state_dict = None
    te_lora_state_dict = None
    if config.unet_peft:
        unet_lora_state_dict = get_peft_model_state_dict(
            _unwrap_objects(unet), adapter_name="default"
        )
    if config.text_encoder_peft:
        te_lora_state_dict = get_peft_model_state_dict(
            _unwrap_objects(text_encoder), adapter_name="default"
        )

    # Save diffusers
    os.makedirs(f"../train_results/{config.run_name}/output_diffusers", exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=f"../train_results/{config.run_name}/output_diffusers/",
        weight_name=save_file_name,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=te_lora_state_dict,
        safe_serialization=True,
        is_main_process=fabric.is_global_zero,
    )
    # Save kohya
    os.makedirs(f"../train_results/{config.run_name}/output_kohya_ss", exist_ok=True)
    kohya_ss_state_dict = {}
    if unet_lora_state_dict:
        kohya_ss_state_dict |= get_module_kohya_state_dict(
            unet_lora_state_dict, "lora_unet", unet.peft_config["default"].lora_alpha
        )
    if te_lora_state_dict:
        kohya_ss_state_dict |= get_module_kohya_state_dict(
            te_lora_state_dict,
            "lora_te",
            text_encoder.peft_config["default"].lora_alpha,
        )
    save_file(
        kohya_ss_state_dict,
        f"../train_results/{config.run_name}/output_kohya_ss/{save_file_name}",
    )
