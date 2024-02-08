from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import convert_state_dict_to_diffusers, convert_state_dict_to_kohya
import torch

def save_lora_weights(output_dir, file_name, unet=None, text_encoder=None, text_encoder_2=None):
    unet_lora_layers_to_save = None
    text_encoder_lora_layers_to_save = None
    text_encoder_2_lora_layers_to_save = None
    state_dict = {}

    def pack_weights(layers, prefix):
        #print("BOOL!", isinstance(layers, torch.nn.Module))
        layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
        layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
        return layers_state_dict

    if unet:
        unet_lora_layers_to_save = get_peft_model_state_dict(unet)
        unet_lora_layers_to_save = pack_weights(unet_lora_layers_to_save, "unet")
        unet_lora_layers_to_save = convert_state_dict_to_kohya(unet_lora_layers_to_save)
        state_dict.update(unet_lora_layers_to_save)
    if text_encoder:
        text_encoder_lora_layers_to_save = get_peft_model_state_dict(text_encoder)
        text_encoder_lora_layers_to_save = pack_weights(text_encoder_lora_layers_to_save, "text_encoder")
        text_encoder_lora_layers_to_save = convert_state_dict_to_kohya(text_encoder_lora_layers_to_save)
        state_dict.update(text_encoder_lora_layers_to_save)
    if text_encoder_2:
        text_encoder_2_lora_layers_to_save = get_peft_model_state_dict(text_encoder)
        text_encoder_2_lora_layers_to_save = pack_weights(text_encoder_2_lora_layers_to_save, "text_encoder_2")
        text_encoder_2_lora_layers_to_save = convert_state_dict_to_kohya(text_encoder_2_lora_layers_to_save)
        state_dict.update(text_encoder_2_lora_layers_to_save)
    StableDiffusionXLPipeline.write_lora_layers(
        state_dict,
        save_directory=output_dir,
        is_main_process=True,
        save_function=None,
        weight_name=file_name+".safetensors",
        safe_serialization=True
    )