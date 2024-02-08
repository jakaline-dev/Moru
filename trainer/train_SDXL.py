#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image with support for LoRA."""

import itertools
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from config import Config
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_snr,
)
from diffusers.utils import convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from libs.data import MoruDataLoader, MoruDataset
from libs.load_checkpoint import load_sdxl_ckpt
from libs.preprocessing import tokenize_prompt
from libs.sample import sample
from libs.save_model_hook import save_lora_weights
from optimizers import AnyPrecisionAdamW
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from pydantic import ValidationError
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=False,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


logger = get_logger(__name__)


def main(config: Config):
    if not config.name:
        config.name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    else:
        config.name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path(os.getcwd()).parent / "trainer_runs" / config.name
    logging_dir = project_dir / "logs"
    output_dir = project_dir / "output"
    preview_dir = project_dir / "preview"

    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.accelerator.mixed_precision,
        log_with=config.accelerator.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if config.accelerator.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(preview_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load Model
    print("Loading checkpoint...")
    pipe = load_sdxl_ckpt(
        config.checkpoint_path, vae_path=config.vae_path
    )  # , dtype=weight_dtype
    noise_scheduler: DDPMScheduler = DDPMScheduler.from_config(pipe["scheduler_config"])
    tokenizer: CLIPTokenizer = pipe["tokenizer"]
    tokenizer_2: CLIPTokenizer = pipe["tokenizer_2"]
    text_encoder: CLIPTextModel = pipe["text_encoder"]
    text_encoder_2: CLIPTextModelWithProjection = pipe["text_encoder_2"]
    unet: UNet2DConditionModel = pipe["unet"]
    vae: AutoencoderKL = pipe["vae"]

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(
        # accelerator.device,
        dtype=weight_dtype if weight_dtype != torch.float16 else torch.float32,
    )
    unet.to(dtype=weight_dtype)  # accelerator.device
    text_encoder.to(dtype=weight_dtype)
    text_encoder_2.to(dtype=weight_dtype)

    if config.xformers:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    if config.lr_unet > 0:
        unet.add_adapter(LoraConfig(**config.peft_unet.model_dump()))

    if config.lr_text_encoder > 0:
        text_encoder.add_adapter(LoraConfig(**config.peft_text_encoder.model_dump()))

    if config.lr_text_encoder_2 > 0:
        text_encoder_2.add_adapter(
            LoraConfig(**config.peft_text_encoder_2.model_dump())
        )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_2))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
        if config.lr_unet > 0:
            unet_state_dict = {
                f'{k.replace("unet.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("unet.")
            }
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(
                unet_, unet_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        if config.lr_text_encoder > 0:
            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_
            )

        if config.lr_text_encoder_2 > 0:
            _set_state_dict_into_text_encoder(
                lora_state_dict,
                prefix="text_encoder_2.",
                text_encoder=text_encoder_two_,
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if config.accelerator.mixed_precision == "fp16":
            models = []
            if config.lr_unet > 0:
                models.extend([unet_])
            if config.lr_text_encoder > 0:
                models.extend([text_encoder_one_])
            if config.lr_text_encoder_2 > 0:
                models.extend([text_encoder_two_])
            cast_training_params(models, dtype=torch.float32)

    # accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if config.gradient_checkpointing:
        if config.lr_unet > 0:
            unet.enable_gradient_checkpointing()
        if config.lr_text_encoder > 0:
            text_encoder.gradient_checkpointing_enable()
        if config.lr_text_encoder_2 > 0:
            text_encoder_2.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(torch.device("cuda"))
        if capability[0] >= 8:
            print("TF32 enabled")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

    # if args.scale_lr:
    #     args.learning_rate = (
    #         args.learning_rate
    #         * args.gradient_accumulation_steps
    #         * args.train_batch_size
    #         * accelerator.num_processes
    #     )

    # Make sure the trainable params are in float32.
    if config.accelerator.mixed_precision == "fp16":
        models = []
        if config.lr_unet > 0:
            models.extend([unet])
        if config.lr_text_encoder > 0:
            models.extend([text_encoder])
        if config.lr_text_encoder_2 > 0:
            models.extend([text_encoder_2])
        cast_training_params(models, dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    match config.optimizer.type:
        case "AdamW8bit":
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        case "AdamW":
            optimizer_class = torch.optim.AdamW
        case "AnyPrecisionAdamW":
            optimizer_class = AnyPrecisionAdamW

    # Optimizer creation
    params_to_optimize = []
    if config.lr_unet > 0:
        params_to_optimize += [
            {"params": p, "lr": config.lr_unet}
            for p in unet.parameters()
            if p.requires_grad
        ]
    if config.lr_text_encoder > 0:
        params_to_optimize += [
            {"params": p, "lr": config.lr_text_encoder}
            for p in text_encoder.parameters()
            if p.requires_grad
        ]
    if config.lr_text_encoder_2 > 0:
        params_to_optimize += [
            {"params": p, "lr": config.lr_text_encoder_2}
            for p in text_encoder_2.parameters()
            if p.requires_grad
        ]
    optimizer = optimizer_class(
        params_to_optimize,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
        eps=config.optimizer.eps,
    )

    dataset = MoruDataset(
        random_crop=config.dataset.random_crop,
        random_flip=config.dataset.random_flip,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
    )
    dataset.load_local_folder(config.dataset.local_path)
    if config.cache_vae:
        with accelerator.autocast():
            dataset.cache_vae(vae=vae, device=accelerator.device)
    dataloader = MoruDataLoader(
        dataset, seed=config.seed, **config.dataloader.model_dump()
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / config.gradient_accumulation_steps
    )
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.num_warmup_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    # if config.lr_unet > 0:
    unet = accelerator.prepare_model(unet)
    # if config.lr_text_encoder > 0:
    text_encoder = accelerator.prepare_model(text_encoder)
    # if config.lr_text_encoder_2 > 0:
    text_encoder_2 = accelerator.prepare_model(text_encoder_2)

    (
        optimizer,
        dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        optimizer,
        dataloader,
        lr_scheduler,
    )

    if not config.cache_vae:
        vae.to(device=accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / config.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(
        config.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(config))

    # Train!
    total_batch_size = (
        config.dataloader.batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )
    if config.sample_strategy == "epoch":
        config.sample_steps = num_update_steps_per_epoch
    elif config.sample_strategy == "no":
        config.sample_steps = -1

    if config.save_strategy == "epoch":
        config.save_steps = num_update_steps_per_epoch
    elif config.save_strategy == "only_last":
        config.save_steps = -1

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {config.dataloader.batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint != "latest":
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = os.listdir(args.output_dir)
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None

    #     if path is None:
    #         accelerator.print(
    #             f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
    #         )
    #         args.resume_from_checkpoint = None
    #         initial_global_step = 0
    #     else:
    #         accelerator.print(f"Resuming from checkpoint {path}")
    #         accelerator.load_state(os.path.join(args.output_dir, path))
    #         global_step = int(path.split("-")[1])

    #         initial_global_step = global_step
    #         first_epoch = global_step // num_update_steps_per_epoch

    # else:
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if config.lr_unet > 0:
        unet.train()
    if config.lr_text_encoder > 0:
        text_encoder.train()
    if config.lr_text_encoder_2 > 0:
        text_encoder_2.train()

    for epoch in range(first_epoch, config.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with accelerator.autocast():
                    if "latent_values" in batch:
                        model_input = DiagonalGaussianDistribution(
                            batch["latent_values"]  # .to(dtype=vae.dtype)
                        ).sample()
                    else:
                        model_input = vae.encode(
                            batch["pixel_values"]
                        ).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # if config.vae_path is None:
                #    model_input = model_input.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                if config.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1),
                        device=model_input.device,
                    )

                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(
                    model_input, noise, timesteps
                )

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left, target_size):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    add_time_ids = list(
                        original_size + crops_coords_top_left + target_size
                    )
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [
                        compute_time_ids(s, c, t)
                        for s, c, t in zip(
                            batch["original_sizes"],
                            batch["crop_top_lefts"],
                            batch["target_sizes"],
                        )
                    ]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder, text_encoder_2],
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[
                        batch["input_ids"],
                        batch["input_ids_2"],
                    ],
                )
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                with accelerator.autocast():
                    if config.min_snr is None:
                        loss = F.mse_loss(
                            # model_pred.float(), noise.float(), reduction="mean"
                            model_pred,
                            noise,
                            reduction="mean",
                        )
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack(
                            [snr, config.min_snr * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        mse_loss_weights = mse_loss_weights / snr

                        loss = F.mse_loss(
                            model_pred.float(), noise.float(), reduction="none"
                        )
                        loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )
                        loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(config.dataloader.batch_size)
                ).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(
                            itertools.chain(*[p["params"] for p in params_to_optimize])
                        ),
                        1.0,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if (
                    accelerator.is_main_process
                    and config.save_steps
                    and global_step % config.save_steps == 0
                ):
                    save_lora_weights(
                        output_dir,
                        f"checkpoint-{global_step}",
                        unet=unet if config.lr_unet > 0 else None,
                        text_encoder=text_encoder
                        if config.lr_text_encoder > 0
                        else None,
                        text_encoder_2=text_encoder_2
                        if config.lr_text_encoder_2 > 0
                        else None,
                    )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            if (
                accelerator.is_main_process
                and config.sample_steps
                and global_step % config.sample_steps == 0
            ):
                with accelerator.autocast():
                    sample(
                        f"{preview_dir}/{global_step}",
                        config,
                        device=accelerator.device,
                        unet=unwrap_model(unet),
                        text_encoder=unwrap_model(text_encoder),
                        text_encoder_2=unwrap_model(text_encoder_2),
                        vae=vae,
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        noise_scheduler_config=noise_scheduler.config,
                    )

                if config.lr_unet > 0:
                    unet.train()
                if config.lr_text_encoder > 0:
                    text_encoder.train()
                if config.lr_text_encoder_2 > 0:
                    text_encoder_2.train()

            if global_step >= config.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     if config.lr_unet > 0:
    #         unet = unwrap_model(unet)
    #         unet_lora_state_dict = convert_state_dict_to_diffusers(
    #             get_peft_model_state_dict(unet)
    #         )
    #     else:
    #         unet_lora_state_dict = None

    #     if config.lr_text_encoder > 0:
    #         text_encoder = unwrap_model(text_encoder)
    #         text_encoder_lora_layers = convert_state_dict_to_diffusers(
    #             get_peft_model_state_dict(text_encoder)
    #         )
    #     else:
    #         text_encoder_lora_layers = None

    #     if config.lr_text_encoder_2 > 0:
    #         text_encoder_2 = unwrap_model(text_encoder_2)
    #         text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
    #             get_peft_model_state_dict(text_encoder_2)
    #         )
    #     else:
    #         text_encoder_2_lora_layers = None

    #     StableDiffusionXLPipeline.save_lora_weights(
    #         save_directory=output_dir,
    #         unet_lora_layers=unet_lora_state_dict,
    #         text_encoder_lora_layers=text_encoder_lora_layers,
    #         text_encoder_2_lora_layers=text_encoder_2_lora_layers,
    #     )

    #     del unet
    #     del text_encoder
    #     del text_encoder_2
    #     del text_encoder_lora_layers
    #     del text_encoder_2_lora_layers
    #     torch.cuda.empty_cache()

    accelerator.end_training()


def load_and_validate_json(file_path: str) -> Config:
    with open(file_path, "r") as file:
        data = json.load(file)  # Load and parse the JSON file

    try:
        config = Config(**data)  # Validate and create a Config instance
        return config  # Return the validated config
    except ValidationError as e:
        print("Validation Error:", e.json())
        raise


if __name__ == "__main__":
    file_path = "config_example.json"
    config = load_and_validate_json(file_path)
    main(config)
