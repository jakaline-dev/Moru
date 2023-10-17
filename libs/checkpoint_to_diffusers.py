import torch
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_vae_checkpoint,
    convert_ldm_unet_checkpoint,
)


def checkpoint_to_diffusers(checkpoint_path, type="SD1", return_vae=True):
    file_path = Path(checkpoint_path)
    if not file_path.is_file():
        raise Exception("Checkpoint file does not exist")
    if file_path.suffix == ".safetensors":
        from safetensors.torch import load_file as safe_load

        checkpoint = safe_load(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = checkpoint["state_dict"]

    # SD1
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # scheduler
    scheduler = DDIMScheduler(
        beta_end=0.0120,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )
    scheduler.register_to_config(clip_sample=False)
    config = dict(scheduler.config)
    config["skip_prk_steps"] = True
    scheduler = PNDMScheduler.from_config(config)

    # unet
    unet_config = {
        "sample_size": 64,
        "in_channels": 4,
        "down_block_types": (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        "block_out_channels": (320, 640, 1280, 1280),
        "layers_per_block": 2,
        "cross_attention_dim": 768,
        "attention_head_dim": 8,
        "use_linear_projection": False,
        "class_embed_type": None,
        "addition_embed_type": None,
        "addition_time_embed_dim": None,
        "projection_class_embeddings_input_dim": None,
        "transformer_layers_per_block": 1,
        "out_channels": 4,
        "up_block_types": (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        "upcast_attention": None,
    }
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config)
    unet = UNet2DConditionModel(**unet_config)
    unet.load_state_dict(converted_unet_checkpoint)

    # vae
    if return_vae:
        vae_config = {
            "sample_size": 512,
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": (
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            "up_block_types": (
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            "block_out_channels": (128, 256, 512, 512),
            "latent_channels": 4,
            "layers_per_block": 2,
            "scaling_factor": 0.18215,
        }
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)
        vae_config["scaling_factor"] = 0.18215
        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(converted_vae_checkpoint)
    else:
        vae = None
    # text_encoder
    text_encoder_config = {
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dropout": 0.0,
        "eos_token_id": 2,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77,
        "model_type": "clip_text_model",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 1,
        "projection_dim": 768,
        "transformers_version": "4.34.0",
        "vocab_size": 49408,
    }

    text_model = CLIPTextModel(CLIPTextConfig(**text_encoder_config))
    keys = list(checkpoint.keys())
    text_model_dict = {}

    remove_prefixes = [
        "cond_stage_model.transformer",
        "conditioner.embedders.0.transformer",
    ]
    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                key_fix = "text_model." + str(key[len(prefix + ".") :])
                text_model_dict[key_fix] = checkpoint[key]

    if not (
        hasattr(text_model, "embeddings")
        and hasattr(text_model.embeddings.position_ids)
    ):
        text_model_dict.pop("text_model.embeddings.position_ids", None)

    text_model.load_state_dict(text_model_dict)
    return vae, text_model, tokenizer, unet, scheduler
