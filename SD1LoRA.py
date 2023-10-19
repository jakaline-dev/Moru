import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import lightning as L
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.fabric.wrappers import _unwrap_objects

from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.training_utils import compute_snr

from diffusers.models.vae import DiagonalGaussianDistribution
import safetensors

from peft import LoraConfig, inject_adapter_in_model, get_peft_model, AdaLoraConfig


def get_training_parameters(config, unet, text_encoder):
    parameters = []
    if config.peft.unet:
        lora_unet_config = LoraConfig(**config.peft.unet.parameters)
        unet = inject_adapter_in_model(lora_unet_config, unet)
        parameters += [
            {
                "params": [
                    p
                    for n, p in unet.named_parameters()
                    if (p.requires_grad and "bias" not in n)
                ],
                "lr": config.peft.unet.lr,
            },
            {
                "params": [
                    p
                    for n, p in unet.named_parameters()
                    if (p.requires_grad and "bias" in n)
                ],
                "lr": config.peft.unet.lr,
                "weight_decay": 0.0,
            },
        ]
    if config.peft.te:
        lora_te_config = LoraConfig(**config.peft.te.parameters)
        text_encoder = inject_adapter_in_model(lora_te_config, text_encoder)
        parameters += [
            {
                "params": [
                    p
                    for n, p in text_encoder.named_parameters()
                    if (p.requires_grad and "bias" not in n)
                ],
                "lr": config.peft.te.lr,
            },
            {
                "params": [
                    p
                    for n, p in text_encoder.named_parameters()
                    if (p.requires_grad and "bias" in n)
                ],
                "lr": config.peft.te.lr,
                "weight_decay": 0.0,
            },
        ]
    return parameters, unet, text_encoder


def train(
    config,
    fabric: L.Fabric,
    dataloader: DataLoader,
    total_steps: int,
    tokenizer,
    noise_scheduler,
    text_encoder,
    vae,
    unet,
    optimizer,
    lr_scheduler,
):
    current_epoch = 0
    current_step = 0
    pbar = None
    if fabric.is_global_zero:
        pbar = tqdm(total=total_steps, desc="")
    while True:
        current_epoch += 1
        pbar.set_description(f"Epoch {current_epoch}")
        for idx, batch in enumerate(dataloader):
            current_step += 1
            is_accumulating = (idx + 1) % config.trainer.grad_accum_steps != 0
            with fabric.autocast():
                if "latent_values" in batch:
                    latents = DiagonalGaussianDistribution(
                        batch["latent_values"].to(dtype=vae.dtype)
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
                if config.trainer.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.trainer.noise_offset * torch.randn(
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
                    encoder_outputs.hidden_states[-config.trainer.clip_skip].to(
                        dtype=latents.dtype
                    )
                )
                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                # del noisy_latents
                # del encoder_hidden_states
                if not config.trainer.snr_gamma:
                    loss = F.mse_loss(model_pred, noise, reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    # print(snr.item(), timesteps.item())
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                config.trainer.snr_gamma * torch.ones_like(timesteps),
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
                # pbar.set_postfix(loss = loss.item())
            fabric.backward(loss / config.trainer.grad_accum_steps)
            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()

            if (
                config.logging.save.every == "step"
                and current_step % config.logging.save.value == 0
            ):
                save_lora_checkpoint(config, unet, text_encoder, current_step)
            if (
                config.logging.sample.every == "step"
                and current_step % config.logging.sample.value == 0
            ):
                sample_images(
                    config,
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
            lr_scheduler.step()

        if (
            config.trainer.max_train.method == "epoch"
            and current_epoch >= config.trainer.max_train.value
        ):
            return
        if (
            config.logging.save.every == "epoch"
            and current_epoch % config.logging.save.value == 0
        ):
            save_lora_checkpoint(config, unet, text_encoder, current_epoch)
        if (
            config.logging.sample.every == "epoch"
            and current_epoch % config.logging.sample.value == 0
        ):
            sample_images(
                config,
                fabric,
                tokenizer,
                noise_scheduler,
                text_encoder,
                vae,
                unet,
                current_epoch,
            )


def save_lora_checkpoint(config, unet, text_encoder, current_iter=None):
    state_dict = {}
    if config.peft.unet:
        for name, params in _unwrap_objects(unet).named_parameters():
            if params.requires_grad:
                state_dict[name] = params
    if config.peft.te:
        for name, params in _unwrap_objects(text_encoder).named_parameters():
            if params.requires_grad:
                state_dict[name] = params
    if current_iter:
        save_file_name = (
            f"{config.name}_{current_iter}_{config.logging.save.every}.safetensors"
        )
    else:
        save_file_name = f"{config.name}.safetensors"

    os.makedirs(f"runs/{config.run_name}/output", exist_ok=True)
    safetensors.torch.save_file(
        state_dict,
        f"runs/{config.run_name}/output/{save_file_name}",
        metadata={"format": "pt"},
    )


@torch.inference_mode()
def sample_images(
    config,
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
    os.makedirs(f"runs/{config.run_name}/samples", exist_ok=True)
    if current_iter:
        save_file_name = f"runs/{config.run_name}/samples/{current_iter}_{config.logging.sample.every}.png"
    else:
        save_file_name = f"runs/{config.run_name}/samples/final.png"
    if config.datapipe.preprocess.cache_vae_outputs:
        vae.to(fabric.device)
    pipeline = StableDiffusionPipeline(
        tokenizer=tokenizer,
        scheduler=EulerDiscreteScheduler.from_config(noise_scheduler.config),
        text_encoder=text_encoder,
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
        image = pipeline(generator=generator, **config.logging.sample.pipeline).images[
            0
        ]
        image.save(save_file_name)
        del image
        del pipeline
    if config.datapipe.preprocess.cache_vae_outputs:
        vae.to("cpu")
    torch.cuda.empty_cache()
