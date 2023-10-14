# Moru Trainer

Diffusers meets Lightning Fabric.

Friendship ended with Accelerate - Now Lightning Fabric is my best friend.

Not production-ready - not ready at all atm.
(Tested in python 3.11, pytorch 2.1.0, CUDA 12.1)

## Features

- SD1 LoRA Training

## Usage

Currently only SD1 LoRA training available.

```bash
python SD1LoRA.py --config config-example.yaml
```

### Config

```yaml
seed: 42
base:
  checkpoint_path: # Your base checkpoint.
  type: SD1
  clip_skip: 2

trainer:
  accelerator: auto
  strategy: auto
  devices: auto
  # bf16-mixed: mixed precision training (bf16 <> fp32)
  # bf16-true: full bf16 training
  # Only use those two options
  # If your GPU is not above NVIDIA Ampere, cry or get a runpod instance
  precision: bf16-true
  batch_size: 1
  grad_accum_steps: 1
  # Pytorch 2.1 default attention is faster than xformers in windows (Could change with implementation of Flash Attention 2)
  use_xformers: false
  name: # Name of the training LoRA
  max_train:
    method: "epoch"
    value: 10
  save_every:
    method: "epoch"
    value: 1
  sample_every:
    method: "epoch"
    value: 0
    num_images: 0

model:
  type: SD1LoRA
  clip_skip: ${base.clip_skip}
  noise_offset: 0.05
  snr_gamma: 5
  lora_unet:
    enable_train: true
    rank: 16
    network_alpha: 8
    optimizer:
      # AdamW: The AdamW you know
      # AdamW8bit: bitsandbytes optimizer is supported, but is not that useful for LoRA training in SD1
      name: AdamW
      init_args:
        lr: 0.0005
    lr_scheduler:
      name: cosine_with_restarts
      init_args:
        num_cycles: 3
        num_warmup_steps: 100
  lora_te:
    enable_train: true
    rank: 16
    network_alpha: 8
    optimizer:
      name: AdamW
      init_args:
        lr: 0.00005
    lr_scheduler:
      name: cosine_with_restarts
      init_args:
        num_cycles: 3
        num_warmup_steps: 100

preprocess:
  folder_path: #Your dataset. A folder with images and captions(.txt) file.
  parameters:
    # 1 chunk = 64 x 64 pixels, 64 chunk = 512 x 512 pixels, 256 chunk = 1024 x 1024 pixels
    # If images are smaller than min_chunk area, remove it from dataset
    # If images are bigger than max_chunk area, resize them until it fits.
    max_chunk: 96
    min_chunk: 32

dataset:
  # Bucketing with only batch_size=1 supported
  # Will be replaced by complete bucketing implementation soon
  type: batch1
  parameters:
    force_upscale: false
    random_crop: true
    random_flip: true
    # shuffle tags / cache currently not supported
    shuffle_tags: false
    cache_latents: false
    cache_text_embeddings: false

dataloader:
  batch_size: ${trainer.batch_size}
  pin_memory: true
  num_workers: 1
  persistent_workers: true
```

### Remarks about 'network_alpha' value

In the LoRA paper, the authors scale the weight updates by a factor of network_alpha divided by the LoRA rank. The introduction of this alpha value serves to normalize tests conducted with various ranks, since the magnitude of the weight updates is proportional to these ranks. By establishing a standard alpha value of 8, the authors are able to facilitate a fair comparison of LoRAs with different ranks, ensuring consistency in the size of the weight updates throughout training.

TL;DR: Quoting the paper,

> ...tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately.

## Todo

### Todo (ASAP)

- Image Bucketing
- SDXL Finetuning
- Easy Installers
- SDXL LoRAs
- Textual Inversion
- Docs! (Docs.)

### Future Endeavours

- Image Padding / Masking
- LoRA-FA
- P+ (TI embeds with layers)
- NeTI (TI embeds with timesteps)
- Würstchen v3 (When it's up)

## Shoutouts

This trainer was built from the shoulders of many giants. Also, just giving a shoutout who led me to this rabbit hole.

- [Diffusers](https://huggingface.co/docs/diffusers/index) from huggingface
- [Kohya Trainer](https://github.com/kohya-ss/sd-scripts)
- And [Stability AI](https://github.com/Stability-AI/generative-models)
