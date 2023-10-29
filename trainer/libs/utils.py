import os

import torch
from diffusers import StableDiffusionPipeline
from libs.convert_LoRA import get_module_kohya_state_dict
from lightning.fabric.wrappers import _unwrap_objects
from peft import LoraConfig, get_peft_model
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


def print_trainable_parameters(model):
    def format_large_number(num):
        if num >= 10**9:  # Billions
            return "{:.2f}B".format(num / 10**9)
        elif num >= 10**6:  # Millions
            return "{:.2f}M".format(num / 10**6)
        elif num >= 10**3:  # Millions
            return "{:.2f}K".format(num / 10**3)
        else:
            return str(num)

    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {format_large_number(trainable_params)} || all params: {format_large_number(all_param)} || trainable%: {100 * trainable_params / all_param}%"
    )


def get_training_parameters_lora(config, unet, text_encoder):
    parameters = []
    if config.unet_peft:
        for unet_peft in config.unet_peft:
            unet_lora_config = LoraConfig(**unet_peft.parameters)
            unet.add_adapter(unet_lora_config)
            # unet_lora_layer_names = list(get_peft_model(unet, unet_lora_config).keys())
            params = get_peft_model(unet, unet_lora_config).parameters()
            print("U-NET trainable parameters:")
            print_trainable_parameters(unet)
            parameters += [
                {"params": params, "lr": unet_peft.lr, "weight_decay": 0.0},
            ]
    if config.text_encoder_peft:
        for te_peft in config.text_encoder_peft:
            te_lora_config = LoraConfig(**te_peft.parameters)
            text_encoder.add_adapter(te_lora_config)
            params = get_peft_model(text_encoder, te_lora_config).parameters()
            print("Text Encoder trainable parameters:")
            print_trainable_parameters(text_encoder)
            parameters += [
                {"params": params, "lr": te_peft.lr, "weight_decay": 0.0},
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
