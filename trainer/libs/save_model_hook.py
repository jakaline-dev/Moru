from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_diffusers, convert_state_dict_to_kohya
import os

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_lora_weights(output_dir, file_name, unet=None, text_encoder=None, text_encoder_2=None):
    unet_lora_layers_to_save = None
    text_encoder_lora_layers_to_save = None
    text_encoder_2_lora_layers_to_save = None

    if unet:
        unet_lora_layers_to_save = get_peft_model_state_dict(unet)
        unet_lora_layers_to_save = convert_state_dict_to_diffusers(unet_lora_layers_to_save)
    if text_encoder:
        text_encoder_lora_layers_to_save = get_peft_model_state_dict(text_encoder)
        text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(text_encoder_lora_layers_to_save)
    if text_encoder_2:
        text_encoder_2_lora_layers_to_save = get_peft_model_state_dict(text_encoder_2)
        text_encoder_2_lora_layers_to_save = convert_state_dict_to_diffusers(text_encoder_2_lora_layers_to_save)
    StableDiffusionXLPipeline.save_lora_weights(
        output_dir,
        unet_lora_layers=unet_lora_layers_to_save,
        text_encoder_lora_layers=text_encoder_lora_layers_to_save,
        text_encoder_2_lora_layers=text_encoder_2_lora_layers_to_save,
        weight_name=file_name+".safetensors"
    )