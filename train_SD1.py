import os
import time
import torch
import torch.nn.functional as F

import lightning as L
from lightning.fabric import Fabric
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.fabric.wrappers import _unwrap_objects

from diffusers.utils.import_utils import is_xformers_available
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.models.vae import DiagonalGaussianDistribution
from transformers import CLIPTextModel
from tqdm.auto import tqdm

from libs.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)
from libs.preprocessing import (
    cache_te_outputs,
    load_data,
    captions_to_tokens,
    cache_vae_outputs,
)
from libs.dataset import MoruDataset
from libs.bucket_dataloader import get_bucket_dataloader
from libs.utils import load_optimizer

from configs.SD1Config import SD1Config
from SD1LoRA import get_training_parameters, save_lora_checkpoint


def main(config: SD1Config):
    if config.seed:
        L.seed_everything(config.seed)
    fabric = Fabric(**config.fabric)
    with fabric.init_module():
        pipe = download_from_original_stable_diffusion_ckpt(
            config.paths.base_checkpoint_path,
            from_safetensors=config.paths.base_checkpoint_path.endswith(".safetensors"),
            load_safety_checker=False,
            local_files_only=True,
            device=fabric.device,
        )
        noise_scheduler = pipe.scheduler
        tokenizer = pipe.tokenizer
        text_encoder: CLIPTextModel = pipe.text_encoder
        unet = pipe.unet
        vae = pipe.vae

    vae.requires_grad_(False)
    for param in unet.parameters():
        param.requires_grad_(False)
    for param in text_encoder.parameters():
        param.requires_grad_(False)

    if config.trainer.use_xformers and is_xformers_available():
        import xformers

        unet.enable_xformers_memory_efficient_attention()
        print("xformers enabled!")

    # preprocess
    data_list = load_data(config.paths.train_data_path, **config.preprocess)
    assert len(data_list) > 0, "Empty folder or incorrect path"
    # Cache VAE latent output
    if config.trainer.cache_vae_outputs:
        fabric.print("Caching VAE outputs")
        with fabric.autocast():
            data_list = cache_vae_outputs(data_list, vae, fabric.device)

    # Cache Tokenizer
    fabric.print("Caching Tokenizer")
    data_list = captions_to_tokens(data_list, tokenizer)

    # Cache Text Encoder output
    if config.trainer.cache_te_outputs:
        fabric.print("Caching Text Encoder outputs")
        with fabric.autocast():
            data_list = cache_te_outputs(
                data_list, text_encoder, fabric.device, config.text_encoder.clip_skip
            )

    # Load Dataset
    dataset = MoruDataset(data_list, **config.dataset)
    dataloader = get_bucket_dataloader(dataset, **config.dataloader)
    fabric.print(
        {key: len(value) for key, value in dataloader.batch_sampler.buckets.items()}
    )

    dataloader = fabric.setup_dataloaders(dataloader)

    if config.trainer.max_train.method == "step":
        total_steps = config.trainer.max_train.value
    elif config.trainer.max_train.method == "epoch":
        total_steps = (
            len(dataloader) // config.trainer.grad_accum_steps
        ) * config.trainer.max_train.value
    else:
        raise Exception("max_train.method is either 'step' or 'epoch'")

    trainable_parameters = []
    p, unet, text_encoder = get_training_parameters(config, unet, text_encoder)
    trainable_parameters += p
    optimizer = load_optimizer(config.optimizer.name)(
        trainable_parameters, **config.optimizer.init_args
    )
    if config.unet.train:
        unet = fabric.setup(unet)
    if config.text_encoder.train:
        text_encoder = fabric.setup(text_encoder)
    optimizer = fabric.setup_optimizers(optimizer)

    lr_scheduler = get_scheduler(
        config.lr_scheduler.name,
        optimizer=optimizer,
        num_training_steps=total_steps,
        **config.lr_scheduler.init_args,
    )

    unet.to(fabric.device)
    if not config.trainer.cache_vae_outputs:
        vae.to(fabric.device)
    if not config.text_encoder.train and not config.trainer.cache_te_outputs:
        text_encoder.to(fabric.device)

    train_time = time.perf_counter()

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
                if config.unet.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.unet.noise_offset * torch.randn(
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

                if "text_embeddings" in batch:
                    encoder_hidden_states = batch["text_embeddings"]
                else:
                    # Get the text embedding for conditioning
                    encoder_outputs = text_encoder(
                        batch["input_ids"], output_hidden_states=True
                    )

                    encoder_hidden_states = text_encoder.text_model.final_layer_norm(
                        encoder_outputs.hidden_states[
                            -config.text_encoder.clip_skip
                        ].to(dtype=latents.dtype)
                    )
                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                # del noisy_latents
                # del encoder_hidden_states
                if not config.unet.snr_gamma:
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
                                config.unet.snr_gamma * torch.ones_like(timesteps),
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
                save_lora_checkpoint(config, fabric, unet, text_encoder, current_step)
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
            save_lora_checkpoint(config, fabric, unet, text_encoder, current_epoch)
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
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


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

    CPU_TE = False

    if config.trainer.cache_vae_outputs:
        vae.to(fabric.device)

    if text_encoder.device == "cpu":
        CPU_TE = True
        text_encoder.to(fabric.device)

    pipeline = StableDiffusionPipeline(
        tokenizer=tokenizer,
        scheduler=EulerDiscreteScheduler.from_config(noise_scheduler.config),
        text_encoder=_unwrap_objects(text_encoder)
        if config.text_encoder.train
        else text_encoder,
        unet=_unwrap_objects(unet) if config.unet.train else unet,
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
            generator=generator,
            clip_skip=config.text_encoder.clip_skip,
            **config.logging.sample.pipeline,
        ).images[0]
        image.save(save_file_name)
        del image
        del pipeline

    if config.trainer.cache_vae_outputs:
        vae.to("cpu")
    if CPU_TE:
        text_encoder.to("cpu")
    torch.cuda.empty_cache()
