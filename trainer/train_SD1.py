import os
import time

import lightning as L
import torch
import torch.nn.functional as F
from configs.SD1Config import SD1Config
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils.import_utils import is_xformers_available
from libs.bucket_dataloader import get_bucket_dataloader
from libs.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from libs.dataset import MoruDataset
from libs.preprocessing import (
    add_caption_template,
    cache_te_outputs,
    cache_tokenizer_output,
    cache_vae_outputs,
    load_data,
)
from libs.utils import (
    get_te_lora_parameters,
    get_unet_lora_parameters,
    init_textual_inversion,
    load_optimizer,
    save_checkpoint_lora,
)
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.seed import isolate_rng
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def main(config: SD1Config):
    if config.seed:
        L.seed_everything(config.seed)
    fabric = Fabric(**config.fabric)
    with fabric.init_module():
        pipe = download_from_original_stable_diffusion_ckpt(
            config.paths.base_checkpoint_path,
            from_safetensors=config.paths.base_checkpoint_path.endswith(".safetensors"),
            local_files_only=True,
            device=fabric.device,
        )
        noise_scheduler = pipe.scheduler
        tokenizer: CLIPTokenizer = pipe.tokenizer
        text_encoder: CLIPTextModel = pipe.text_encoder
        unet: UNet2DConditionModel = pipe.unet
        vae: AutoencoderKL = pipe.vae

    # Freeze All
    for param in vae.parameters():
        param.requires_grad_(False)
    for param in unet.parameters():
        param.requires_grad_(False)
    for param in text_encoder.parameters():
        param.requires_grad_(False)

    # gradient checkpointing
    if config.trainer.gradient_checkpointing:
        unet.train()
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    # xformers
    if config.trainer.use_xformers and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        print("xformers enabled!")

    # preprocess
    data_list = load_data(
        config.paths.train_data_path,
        max_chunk=config.preprocess.max_chunk,
        min_chunk=config.preprocess.min_chunk,
    )
    assert len(data_list) > 0, "Empty folder or incorrect path"

    # Cache VAE latent output
    if config.trainer.cache_vae_outputs:
        fabric.print("Caching VAE outputs")
        with fabric.autocast():
            data_list = cache_vae_outputs(data_list, vae, fabric.device)

    # Cache Tokenizer
    fabric.print("Caching Tokenizer")
    if config.preprocess.caption_template:
        data_list = add_caption_template(
            data_list, config.preprocess.caption_template, name=config.name
        )
    data_list = cache_tokenizer_output(data_list, tokenizer)

    # Cache Text Encoder output
    if config.trainer.cache_te_outputs:
        fabric.print("Caching Text Encoder outputs")
        with fabric.autocast():
            data_list = cache_te_outputs(
                data_list, text_encoder, fabric.device, config.trainer.clip_skip
            )

    # Load Dataset
    dataset = MoruDataset(data_list, tokenizer=tokenizer, **config.dataset)
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

    if config.textual_inversion:
        placeholder_token_ids, tokenizer, text_encoder = init_textual_inversion(
            config, tokenizer, text_encoder
        )

    if config.unet_peft:
        unet_peft_params, unet = get_unet_lora_parameters(config, unet)
        trainable_parameters += unet_peft_params

    if config.text_encoder_peft:
        te_peft_params, text_encoder = get_te_lora_parameters(config, text_encoder)
        trainable_parameters += te_peft_params

    optimizer = load_optimizer(config.optimizer.name)(
        trainable_parameters, **config.optimizer.init_args
    )
    if config.is_train_unet:
        unet = fabric.setup(unet)
    if config.is_train_text_encoder:
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
    if not config.is_train_text_encoder and not config.trainer.cache_te_outputs:
        text_encoder.to(fabric.device)

    train_time = time.perf_counter()
    current_epoch = 0
    current_step = 0
    pbar = None
    if fabric.is_global_zero:
        pbar = tqdm(total=total_steps, desc="")
    while not (
        (
            config.trainer.max_train.method == "epoch"
            and current_epoch >= config.trainer.max_train.value
        )
        or (
            config.trainer.max_train.method == "step"
            and current_step >= config.trainer.max_train.value
        )
    ):
        current_epoch += 1
        pbar.set_description(f"Epoch {current_epoch}")
        for idx, batch in enumerate(dataloader):
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

                if "text_embeddings" in batch:
                    encoder_hidden_states = batch["text_embeddings"]
                else:
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
                if not config.trainer.min_snr:
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
                                config.trainer.min_snr * torch.ones_like(timesteps),
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

            fabric.backward(loss / config.trainer.grad_accum_steps)
            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                current_step += 1
                lr_scheduler.step()
                if pbar is not None:
                    pbar.update(1)
                    # pbar.set_postfix(loss = loss.item())
                if (
                    config.trainer.max_train.method == "step"
                    and current_step >= config.trainer.max_train.value
                ):
                    break
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
                    config.logging.save.every == "step"
                    and current_step % config.logging.save.value == 0
                ):
                    save_checkpoint_lora(
                        config, fabric, unet, text_encoder, current_step
                    )

        if (
            config.trainer.max_train.method == "epoch"
            and current_epoch >= config.trainer.max_train.value
        ):
            break
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

        if (
            config.logging.save.every == "epoch"
            and current_epoch % config.logging.save.value == 0
        ):
            save_checkpoint_lora(config, fabric, unet, text_encoder, current_epoch)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    save_checkpoint_lora(config, fabric, unet, text_encoder)
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
    os.makedirs(f"../train_results/{config.run_name}/samples", exist_ok=True)
    if current_iter:
        save_file_name = f"../train_results/{config.run_name}/samples/{current_iter}_{config.logging.sample.every}.png"
    else:
        save_file_name = f"../train_results/{config.run_name}/samples/final.png"

    if config.trainer.cache_vae_outputs:
        vae.to(fabric.device)
    if config.trainer.cache_te_outputs:
        text_encoder.to(fabric.device)

    pipeline = StableDiffusionPipeline(
        tokenizer=tokenizer,
        scheduler=EulerDiscreteScheduler.from_config(noise_scheduler.config),
        text_encoder=_unwrap_objects(text_encoder)
        if config.text_encoder_peft
        else text_encoder,
        unet=_unwrap_objects(unet) if config.unet_peft else unet,
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
            clip_skip=config.trainer.clip_skip,
            **config.logging.sample.pipeline,
        ).images[0]
        image.save(save_file_name)
        del image
        del pipeline

    if config.trainer.cache_vae_outputs:
        vae.to("cpu")
    if config.trainer.cache_te_outputs:
        text_encoder.to("cpu")
    torch.cuda.empty_cache()
