import json
import re
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
)
from diffusers.utils import is_accelerate_available
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

SDXL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "model_configs/SDXL"

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device


def convert_unet_checkpoint(checkpoint, device=None, dtype=None):
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with open(SDXL_CONFIG_PATH / "unet.json") as f:
        unet_config = json.load(f)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, config=unet_config
    )
    with ctx():
        unet = UNet2DConditionModel(**unet_config)
    if is_accelerate_available():
        for param_name, param in converted_unet_checkpoint.items():
            set_module_tensor_to_device(
                unet, param_name, device=device, value=param, dtype=dtype
            )
    else:
        unet.load_state_dict(converted_unet_checkpoint)
    return unet


def convert_vae_checkpoint(checkpoint, device=None, dtype=None):
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with open(SDXL_CONFIG_PATH / "vae.json") as f:
        vae_config = json.load(f)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, config=vae_config)
    with ctx():
        vae = AutoencoderKL(**vae_config)
    if is_accelerate_available():
        for param_name, param in converted_vae_checkpoint.items():
            set_module_tensor_to_device(
                vae, param_name, device=device, value=param, dtype=dtype
            )
    else:
        vae.load_state_dict(converted_vae_checkpoint)
    return vae


def convert_text_encoder_checkpoint(checkpoint, device=None, dtype=None):
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    # with open('../model_configs/SDXL/text_encoder.json') as f:
    #     text_encoder_config = json.load(f)
    config = CLIPTextConfig.from_json_file(SDXL_CONFIG_PATH / "text_encoder.json")
    with ctx():
        text_model = CLIPTextModel(config)

    keys = list(checkpoint.keys())

    text_model_dict = {}

    remove_prefixes = [
        "cond_stage_model.transformer",
        "conditioner.embedders.0.transformer",
    ]

    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                text_model_dict[key[len(prefix + ".") :]] = checkpoint[key]

    if is_accelerate_available():
        for param_name, param in text_model_dict.items():
            set_module_tensor_to_device(
                text_model, param_name, device=device, value=param, dtype=dtype
            )
    else:
        if not (
            hasattr(text_model, "embeddings")
            and hasattr(text_model.embeddings.position_ids)
        ):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_model.load_state_dict(text_model_dict)

    return text_model


def convert_text_encoder_2_checkpoint(checkpoint, device=None, dtype=None):
    textenc_conversion_lst = [
        ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
        ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
        ("ln_final.weight", "text_model.final_layer_norm.weight"),
        ("ln_final.bias", "text_model.final_layer_norm.bias"),
        ("text_projection", "text_projection.weight"),
    ]
    textenc_conversion_map = {x[0]: x[1] for x in textenc_conversion_lst}

    textenc_transformer_conversion_lst = [
        # (stable-diffusion, HF Diffusers)
        ("resblocks.", "text_model.encoder.layers."),
        ("ln_1", "layer_norm1"),
        ("ln_2", "layer_norm2"),
        (".c_fc.", ".fc1."),
        (".c_proj.", ".fc2."),
        (".attn", ".self_attn"),
        ("ln_final.", "transformer.text_model.final_layer_norm."),
        (
            "token_embedding.weight",
            "transformer.text_model.embeddings.token_embedding.weight",
        ),
        (
            "positional_embedding",
            "transformer.text_model.embeddings.position_embedding.weight",
        ),
    ]
    protected = {re.escape(x[0]): x[1] for x in textenc_transformer_conversion_lst}
    textenc_pattern = re.compile("|".join(protected.keys()))
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    config = CLIPTextConfig.from_json_file(SDXL_CONFIG_PATH / "text_encoder_2.json")
    with ctx():
        text_model = CLIPTextModelWithProjection(config)

    keys = list(checkpoint.keys())

    keys_to_ignore = []

    text_model_dict = {}
    prefix = "conditioner.embedders.1.model."
    d_model = 1280

    text_model_dict[
        "text_model.embeddings.position_ids"
    ] = text_model.text_model.embeddings.get_buffer("position_ids")

    for key in keys:
        if key in keys_to_ignore:
            continue
        if key[len(prefix) :] in textenc_conversion_map:
            if key.endswith("text_projection"):
                value = checkpoint[key].T.contiguous()
            else:
                value = checkpoint[key]

            text_model_dict[textenc_conversion_map[key[len(prefix) :]]] = value

        if key.startswith(prefix + "transformer."):
            new_key = key[len(prefix + "transformer.") :]
            if new_key.endswith(".in_proj_weight"):
                new_key = new_key[: -len(".in_proj_weight")]
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )
                text_model_dict[new_key + ".q_proj.weight"] = checkpoint[key][
                    :d_model, :
                ]
                text_model_dict[new_key + ".k_proj.weight"] = checkpoint[key][
                    d_model : d_model * 2, :
                ]
                text_model_dict[new_key + ".v_proj.weight"] = checkpoint[key][
                    d_model * 2 :, :
                ]
            elif new_key.endswith(".in_proj_bias"):
                new_key = new_key[: -len(".in_proj_bias")]
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )
                text_model_dict[new_key + ".q_proj.bias"] = checkpoint[key][:d_model]
                text_model_dict[new_key + ".k_proj.bias"] = checkpoint[key][
                    d_model : d_model * 2
                ]
                text_model_dict[new_key + ".v_proj.bias"] = checkpoint[key][
                    d_model * 2 :
                ]
            else:
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )

                text_model_dict[new_key] = checkpoint[key]

    if is_accelerate_available():
        for param_name, param in text_model_dict.items():
            set_module_tensor_to_device(
                text_model, param_name, device=device, value=param, dtype=dtype
            )
    else:
        if not (
            hasattr(text_model, "embeddings")
            and hasattr(text_model.embeddings.position_ids)
        ):
            text_model_dict.pop("text_model.embeddings.position_ids", None)

        text_model.load_state_dict(text_model_dict)

    return text_model


def load_sdxl_ckpt(file_path, device="cpu", vae_path=None, dtype=None):
    if file_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        checkpoint = safe_load(file_path, device=device)
    else:
        checkpoint = torch.load(file_path, map_location=device)
    # unet
    unet = convert_unet_checkpoint(checkpoint, device=device, dtype=dtype)
    # vae
    if vae_path:
        if vae_path.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load

            vae_checkpoint = safe_load(vae_path, device=device)
        else:
            vae_checkpoint = torch.load(vae_path, map_location=device)
        vae = convert_vae_checkpoint(vae_checkpoint, device=device, dtype=dtype)
    else:
        vae = convert_vae_checkpoint(checkpoint, device=device, dtype=dtype)
    # tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        SDXL_CONFIG_PATH / "tokenizer", local_files_only=True
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        SDXL_CONFIG_PATH / "tokenizer_2", pad_token="!", local_files_only=True
    )
    # text_encoder
    text_encoder = convert_text_encoder_checkpoint(
        checkpoint, device=device, dtype=dtype
    )
    text_encoder_2 = convert_text_encoder_2_checkpoint(
        checkpoint, device=device, dtype=dtype
    )
    # scheduler
    with open(SDXL_CONFIG_PATH / "scheduler.json") as f:
        scheduler_config = json.load(f)
    # scheduler = DDIMScheduler.from_config(scheduler_config)
    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "scheduler_config": scheduler_config,
    }


if __name__ == "__main__":
    load_sdxl_ckpt(
        r"C:/CODE/ComfyUI_windows_portable/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors"
    )
