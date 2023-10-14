# torch.compile()

import os, sys, argparse
from datetime import datetime
import time, math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import lightning as L
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.fabric.loggers import CSVLogger

# from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _unwrap_objects

from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import compute_snr
from diffusers.loaders import (
    PatchedLoraProjection,
)
from diffusers.models.vae import DiagonalGaussianDistribution
from diffusers.models.lora import LoRALinearLayer, LoRACompatibleLinear
from diffusers.optimization import get_scheduler
import safetensors

from libs.custom_dataset import CustomDataset
from libs.checkpoint_to_diffusers import checkpoint_to_diffusers
from libs.utils import cache_vae_outputs, image_folder_to_list, captions_to_tokens
from libs.load_optimizer import load_optimizer
from libs.bucket_dataset import get_bucket_dataloader

config = None
run_name = ""


def set_lora_layer(attn_module_attribute):
    attn_module_attribute.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module_attribute.in_features,
            out_features=attn_module_attribute.out_features,
            rank=config.model.lora_unet.rank,
            network_alpha=config.model.lora_unet.network_alpha,
        )
    )


def replace_module(model, name, new_module):
    name_parts = name.split(".")
    sub_model = model
    for part in name_parts[:-1]:
        sub_model = getattr(sub_model, part)
    setattr(sub_model, name_parts[-1], new_module)


def create_patched_linear_lora(model):
    model = PatchedLoraProjection(
        model,
        network_alpha=config.model.lora_te.network_alpha,
        rank=config.model.lora_te.rank,
    )
    return model


def unet_lora_state_dict(unet):
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                current_lora_layer_sd = lora_layer.state_dict()
                lora_state_dict[
                    f"lora_unet_{name.replace('.', '_')}.alpha"
                ] = torch.tensor(config.model.lora_unet.network_alpha, dtype=unet.dtype)
                for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                    lora_state_dict[
                        f"lora_unet_{name.replace('.', '_')}.lora_{lora_layer_matrix_name}"
                    ] = lora_param

    return lora_state_dict


def text_encoder_lora_state_dict(text_encoder):
    lora_state_dict = {}

    for name, module in text_encoder.named_modules():
        if hasattr(module, "lora_linear_layer"):
            lora_layer = module.lora_linear_layer
            if lora_layer is not None:
                current_lora_layer_sd = lora_layer.state_dict()
                lora_state_dict[
                    f"lora_te_{name.replace('.', '_')}.alpha"
                ] = torch.tensor(
                    config.model.lora_te.network_alpha, dtype=text_encoder.dtype
                )
                for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                    # The matrix name can either be "down" or "up".
                    lora_state_dict[
                        f"lora_te_{name.replace('.', '_')}.lora_{lora_layer_matrix_name}"
                    ] = lora_param
    return lora_state_dict


def setup():
    # plugins = None
    # if quantize is not None and quantize.startswith("bnb."):
    #     if "mixed" in precision:
    #         raise ValueError("Quantization and mixed precision is not supported.")
    #     dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
    #     plugins = BitsandbytesPrecision(quantize[4:], dtype)
    #     precision = None
    # logger = CSVLogger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = Fabric(
        accelerator="cpu"
        if torch.backends.mps.is_available()
        else "auto",  # trainer_config.accelerator,
        strategy=config.trainer.strategy,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        # plugins=None,
        # callbacks=None,
        # loggers=None,
    )
    fabric.launch(main)


def main(fabric: L.Fabric):
    if "seed" in config:
        fabric.seed_everything(config["seed"])

    with fabric.init_module():  # empty_init=(devices > 1)
        (
            vae,
            text_encoder,
            tokenizer,
            unet,
            noise_scheduler,
        ) = checkpoint_to_diffusers(
            config["base"]["checkpoint_path"],
            type=config["base"]["type"],
            return_vae=True,
        )
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if config.trainer.use_xformers and is_xformers_available():
        import xformers

        unet.enable_xformers_memory_efficient_attention()
        print("xformers enabled!")

    # preprocess
    data_list = image_folder_to_list(
        config.preprocess.folder_path, **config.preprocess.parameters
    )
    assert len(data_list) > 0, "Empty folder or incorrect path"
    # Cache VAE latent output
    if config.preprocess.cache_vae_outputs:
        data_list = cache_vae_outputs(data_list, vae)

    data_list = captions_to_tokens(data_list, tokenizer)
    # TODO: Cache Text Encoder output

    # Load Dataset
    dataset = CustomDataset(data_list, **config.dataset)
    if config.bucket.enable:
        dataloader = get_bucket_dataloader(dataset, **config.dataloader)

    dataloader = fabric.setup_dataloaders(dataloader)

    if config.trainer.max_train.method == "step":
        total_steps = config.trainer.max_train.value
    elif config.trainer.max_train.method == "epoch":
        total_steps = (
            math.ceil(len(dataloader) / config.trainer.grad_accum_steps)
            * config.trainer.max_train.value
        )

    lora_unet_params = []
    lora_te_params = []

    if config.model.lora_unet.enable_train:
        for name, module in unet.named_modules():
            if isinstance(module, LoRACompatibleLinear) and "attention" in name:
                set_lora_layer(module)
                lora_unet_params.extend(module.lora_layer.parameters())

        optimizer_unet = load_optimizer(config.model.lora_unet.optimizer.name)(
            lora_unet_params, **config.model.lora_unet.optimizer.init_args
        )
        unet, optimizer_unet = fabric.setup(unet, optimizer_unet)

        lr_scheduler_unet = get_scheduler(
            config.model.lora_unet.lr_scheduler.name,
            optimizer=optimizer_unet,
            num_training_steps=total_steps,
            **config.model.lora_unet.lr_scheduler.init_args,
        )
    else:
        optimizer_unet = None
        lr_scheduler_unet = None

    if config.model.lora_te.enable_train:
        for name, module in text_encoder.named_modules():
            if isinstance(module, torch.nn.modules.linear.Linear):
                new_module = create_patched_linear_lora(module)
                replace_module(text_encoder, name, new_module)
                lora_te_params.extend(new_module.lora_linear_layer.parameters())

        optimizer_te = load_optimizer(config.model.lora_te.optimizer.name)(
            lora_te_params, **config.model.lora_te.optimizer.init_args
        )
        text_encoder, optimizer_te = fabric.setup(text_encoder, optimizer_te)
        lr_scheduler_te = get_scheduler(
            config.model.lora_te.lr_scheduler.name,
            optimizer=optimizer_te,
            num_training_steps=total_steps,
            **config.model.lora_te.lr_scheduler.init_args,
        )
    else:
        optimizer_te = None
        lr_scheduler_te = None

    # load checkpoint

    # strict=False because missing keys due to LoRA weights not contained in state dict
    # load_checkpoint(fabric, model, checkpoint_path, strict=False)

    if "seed" in config:
        fabric.seed_everything(config["seed"] + fabric.global_rank)

    train_time = time.perf_counter()
    train(
        fabric,
        dataloader,
        total_steps,
        tokenizer,
        noise_scheduler,
        text_encoder,
        vae,
        unet,
        optimizer_unet,
        lr_scheduler_unet,
        optimizer_te,
        lr_scheduler_te,
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    save_lora_checkpoint(unet, text_encoder)


def train(
    fabric: L.Fabric,
    dataloader: DataLoader,
    total_steps: int,
    tokenizer,
    noise_scheduler,
    text_encoder,
    vae,
    unet,
    optimizer_unet=None,
    lr_scheduler_unet=None,
    optimizer_te=None,
    lr_scheduler_te=None,
):
    current_epoch = 0
    current_step = 0
    pbar = None
    if fabric.is_global_zero:
        pbar = tqdm(total=total_steps, desc="")

    while True:
        current_epoch += 1
        for idx, batch in enumerate(dataloader):
            current_step += 1
            is_accumulating = (idx + 1) % config.trainer.grad_accum_steps != 0
            with fabric.autocast():
                if "latent_values" in batch:
                    latents = DiagonalGaussianDistribution(
                        batch["latent_values"]
                    ).sample()
                else:
                    latents = vae.encode(
                        batch["pixel_values"].to(
                            dtype=vae.dtype
                        )  # .to(dtype=fabric._precision)
                    ).latent_dist.sample()
                latents *= vae.config.scaling_factor
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if config.model.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.model.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=fabric.device
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=fabric.device,
                    dtype=torch.long,
                )

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_outputs = text_encoder(
                    batch["input_ids"], output_hidden_states=True
                )

                encoder_hidden_states = text_encoder.text_model.final_layer_norm(
                    encoder_outputs.hidden_states[-config.base.clip_skip].to(
                        dtype=latents.dtype
                    )
                )
                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                # del noisy_latents
                # del encoder_hidden_states
                if config.model.snr_gamma is None:
                    loss = F.mse_loss(model_pred, noise, reduction="mean")
                    # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                config.model.snr_gamma * torch.ones_like(timesteps),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(model_pred, noise, reduction="none")
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

            if pbar is not None:
                pbar.update(1)
            fabric.backward(loss / config.trainer.grad_accum_steps)
            if not is_accumulating:
                if optimizer_unet:
                    optimizer_unet.step()
                    optimizer_unet.zero_grad()
                if optimizer_te:
                    optimizer_te.step()
                    optimizer_te.zero_grad()

            if (
                config.trainer.save_every.method == "step"
                and config.trainer.save_every.value > 0
                and current_step % config.trainer.save_every.value == 0
            ):
                save_lora_checkpoint(unet, text_encoder, current_step)
            if (
                config.trainer.sample_every.method == "step"
                and config.trainer.sample_every.value > 0
                and current_step % config.trainer.sample_every.value == 0
            ):
                sample_images(
                    fabric,
                    tokenizer,
                    noise_scheduler,
                    text_encoder,
                    vae,
                    unet,
                    current_step,
                )
            if (
                config.trainer.max_train.method == "step"
                and current_step >= config.trainer.max_train.value
            ):
                return

            if lr_scheduler_unet:
                lr_scheduler_unet.step()
            if lr_scheduler_te:
                lr_scheduler_te.step()

        if (
            config.trainer.save_every.method == "epoch"
            and config.trainer.save_every.value > 0
            and current_epoch % config.trainer.save_every.value == 0
        ):
            save_lora_checkpoint(unet, text_encoder, current_epoch)
        if (
            config.trainer.sample_every.method == "epoch"
            and config.trainer.sample_every.value > 0
            and current_epoch % config.trainer.sample_every.value == 0
        ):
            sample_images(
                fabric,
                tokenizer,
                noise_scheduler,
                text_encoder,
                vae,
                unet,
                current_step
                if config.trainer.sample_every.method == "step"
                else current_epoch,
            )

        if (
            config.trainer.max_train.method == "epoch"
            and current_epoch >= config.trainer.max_train.value
        ):
            return


def save_lora_checkpoint(unet, text_encoder, current_iter=None):
    if current_iter:
        save_file_name = f"{config.name}_{current_iter}_{config.trainer.save_every.method}.safetensors"
    else:
        save_file_name = f"{config.name}.safetensors"

    state_dict = {}

    if config.model.lora_unet.enable_train and config.model.lora_te.enable_train:
        state_dict = {
            **unet_lora_state_dict(_unwrap_objects(unet)),
            **text_encoder_lora_state_dict(_unwrap_objects(text_encoder)),
        }
    elif config.model.lora_unet.enable_train:
        state_dict = unet_lora_state_dict(_unwrap_objects(unet))
    elif config.model.lora_te.enable_train:
        state_dict = text_encoder_lora_state_dict(_unwrap_objects(text_encoder))

    os.makedirs(f"runs/{run_name}/output", exist_ok=True)
    safetensors.torch.save_file(
        state_dict,
        f"runs/{run_name}/output/{save_file_name}",
        metadata={"format": "pt"},
    )


def sample_images(
    fabric: L.Fabric,
    tokenizer,
    noise_scheduler,
    text_encoder,
    vae,
    unet,
    current_iter=None,
):
    # if not fabric.is_global_zero:
    #    return
    os.makedirs(f"runs/{run_name}/samples", exist_ok=True)
    if current_iter:
        save_file_name = f"runs/{run_name}/samples/{current_iter}_{config.trainer.save_every.method}.png"
    else:
        save_file_name = f"runs/{run_name}/samples/final.png"

    with torch.inference_mode():
        pipeline = StableDiffusionPipeline(
            tokenizer=tokenizer,
            scheduler=EulerDiscreteScheduler.from_config(noise_scheduler.config),
            text_encoder=_unwrap_objects(text_encoder),
            unet=_unwrap_objects(unet),
            vae=vae,
            # torch_dtype=vae.dtype,
            feature_extractor=None,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # pipeline = pipeline.to(fabric.device)
        pipeline.set_progress_bar_config(disable=True)
        with isolate_rng():
            generator = (
                torch.Generator(device=fabric.device).manual_seed(config.seed)
                if config.seed
                else None
            )
            image = pipeline(
                prompt="1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt",
                width=512,
                height=640,
                negative_prompt="worse quality, bad quality, ugly, low quality",
                generator=generator,
                num_inference_steps=28,
                clip_skip=config.model.clip_skip,
            ).images[0]
            image.save(save_file_name)
            del image
            del pipeline
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(torch.device("cuda"))
        if capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.allow_tf32 = True
            print("TF32 Enabled")
            torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser(description="Provide configuration file.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        type=str,
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    run_name = f"{config.name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    setup()
