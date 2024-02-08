import os
from pathlib import Path

import torch
from diffusers.utils import convert_state_dict_to_kohya
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file


def save_lora_weights(
    output_dir, file_name, unet=None, text_encoder=None, text_encoder_2=None
):
    unet_lora_layers_to_save = None
    text_encoder_lora_layers_to_save = None
    text_encoder_2_lora_layers_to_save = None
    state_dict = {}

    def pack_weights(layers, prefix):
        layers_weights = (
            layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
        )
        layers_state_dict = {
            f"{prefix}.{module_name}": param
            for module_name, param in layers_weights.items()
        }
        return layers_state_dict

    if unet:
        unet_lora_layers_to_save = get_peft_model_state_dict(unet)
        unet_lora_layers_to_save = pack_weights(unet_lora_layers_to_save, "unet")
        unet_lora_layers_to_save = convert_state_dict_to_kohya(unet_lora_layers_to_save)
        state_dict.update(unet_lora_layers_to_save)
    if text_encoder:
        text_encoder_lora_layers_to_save = get_peft_model_state_dict(text_encoder)
        text_encoder_lora_layers_to_save = pack_weights(
            text_encoder_lora_layers_to_save, "text_encoder"
        )
        text_encoder_lora_layers_to_save = convert_state_dict_to_kohya(
            text_encoder_lora_layers_to_save
        )
        state_dict.update(text_encoder_lora_layers_to_save)
    if text_encoder_2:
        text_encoder_2_lora_layers_to_save = get_peft_model_state_dict(text_encoder_2)
        text_encoder_2_lora_layers_to_save = pack_weights(
            text_encoder_2_lora_layers_to_save, "text_encoder_2"
        )
        text_encoder_2_lora_layers_to_save = convert_state_dict_to_kohya(
            text_encoder_2_lora_layers_to_save
        )
        state_dict.update(text_encoder_2_lora_layers_to_save)

    os.makedirs(output_dir, exist_ok=True)
    save_path = Path(output_dir, file_name + ".safetensors").as_posix()

    save_file(state_dict, save_path, metadata={"format": "pt"})
