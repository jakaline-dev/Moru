import os, sys, time
import argparse
import torch
from datetime import datetime
import lightning as L
from omegaconf import OmegaConf


from lightning.fabric import Fabric
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from libs.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)
from libs.preprocessing import load_data, captions_to_tokens, cache_vae_outputs
from libs.dataset import MoruDataset
from libs.bucket_dataloader import get_bucket_dataloader
from libs.utils import load_optimizer

from configs.PEFTConfig import PEFTConfig
from SD1LoRA import train, get_training_parameters


def main(config):
    if config.seed:
        L.seed_everything(config.seed)
    fabric = Fabric(**config.fabric)
    with fabric.init_module():  # empty_init=(devices > 1)
        pipe = download_from_original_stable_diffusion_ckpt(
            config.pretrained_model.checkpoint_path,
            from_safetensors=config.pretrained_model.checkpoint_path.endswith(
                ".safetensors"
            ),
            load_safety_checker=False,
            local_files_only=True,
            device=fabric.device,
        )
        noise_scheduler = pipe.scheduler
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.to(fabric.device)
        unet = pipe.unet
        vae = pipe.vae.to(fabric.device)

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
    data_list = load_data(
        config.datapipe.path,
        max_chunk=config.datapipe.preprocess.max_chunk,
        min_chunk=config.datapipe.preprocess.min_chunk,
    )
    assert len(data_list) > 0, "Empty folder or incorrect path"
    # Cache VAE latent output
    if config.datapipe.preprocess.cache_vae_outputs:
        fabric.print("Caching VAE outputs")
        with fabric.autocast():
            data_list = cache_vae_outputs(data_list, vae, fabric.device)
    data_list = captions_to_tokens(data_list, tokenizer)
    # TODO: Cache Text Encoder output

    # Load Dataset
    dataset = MoruDataset(data_list, **config.datapipe.dataset)
    dataloader = get_bucket_dataloader(dataset, **config.datapipe.dataloader)
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
    if config.peft.unet:
        unet = fabric.setup(unet)
    if config.peft.te:
        text_encoder = fabric.setup(text_encoder)
    optimizer = fabric.setup_optimizers(optimizer)

    lr_scheduler = get_scheduler(
        config.lr_scheduler.name,
        optimizer=optimizer,
        num_training_steps=total_steps,
        **config.lr_scheduler.init_args,
    )
    train_time = time.perf_counter()
    train(
        config,
        fabric,
        dataloader,
        total_steps,
        optimizer,
        lr_scheduler,
        tokenizer,
        noise_scheduler,
        text_encoder,
        vae,
        unet,
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


if __name__ == "__main__":
    # set environment
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(torch.device("cuda"))
        if capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.allow_tf32 = True
            print("TF32 Enabled")
            torch.set_float32_matmul_precision("medium")

    # argparse
    parser = argparse.ArgumentParser(description="Provide configuration file.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        type=str,
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    # omegaconf
    config = OmegaConf.structured(PEFTConfig)
    # with open("config.yaml", "w") as f:
    #     f.write(OmegaConf.to_yaml(config))
    yaml_config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, yaml_config)
    config.run_name = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{config.name}"
    main(config)
