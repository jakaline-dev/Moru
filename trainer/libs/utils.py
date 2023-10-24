import os

import torch
from diffusers import StableDiffusionPipeline
from libs.convert_LoRA import get_module_kohya_state_dict
from lightning.fabric.wrappers import _unwrap_objects
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file


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


def get_training_parameters_lora(config, unet, text_encoder):
    parameters = []
    if config.unet_peft:
        for unet_peft in config.unet_peft:
            unet_lora_config = LoraConfig(**unet_peft.parameters)
            unet.add_adapter(unet_lora_config)
            parameters += [
                {
                    "params": get_peft_model_state_dict(unet).values(),
                    "lr": unet_peft.lr,
                },
            ]
    if config.text_encoder_peft:
        for te_peft in config.text_encoder_peft:
            te_lora_config = LoraConfig(**te_peft.parameters)
            text_encoder.add_adapter(te_lora_config)
            parameters += [
                {
                    "params": get_peft_model_state_dict(text_encoder).values(),
                    "lr": te_peft.lr,
                },
            ]
    return parameters, unet, text_encoder


def save_checkpoint_lora(config, fabric, unet, text_encoder, current_iter=None):
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
    if config.logging.save.output_diffusers:
        os.makedirs(
            f"../train_results/{config.run_name}/output_diffusers", exist_ok=True
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=f"../train_results/{config.run_name}/output_diffusers/",
            weight_name=save_file_name,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=te_lora_state_dict,
            safe_serialization=True,
            is_main_process=fabric.is_global_zero,
        )
    # Save kohya
    if config.logging.save.output_kohya_ss:
        os.makedirs(
            f"../train_results/{config.run_name}/output_kohya_ss", exist_ok=True
        )
        kohya_ss_state_dict = {}
        if unet_lora_state_dict:
            kohya_ss_state_dict |= get_module_kohya_state_dict(
                unet_lora_state_dict,
                "lora_unet",
                unet.peft_config["default"].lora_alpha,
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
