# Moru Trainer

Diffusers meets Lightning Fabric.

~~Friendship ended with Accelerate - Now Lightning Fabric is my best friend.~~

2x faster than Kohya trainer.

Not production-ready - not ready at all atm.
(Tested in python 3.11, pytorch 2.1.0, CUDA 12.1)

## Features

- SD1 LoRA Training

## Usage

Currently only SD1 LoRA training available.

1. Install necessary dependencies. (Installer WIP)
2. Fill in the name, base.checkpoint_path, preprocess.folder_path in the config-example.yaml file.
3. Run:

```bash
python SD1LoRA.py --config config-example.yaml
```

## Remarks

### Remarks about 'network_alpha' value

In the LoRA paper, the authors scale the weight updates by a factor of network_alpha divided by the LoRA rank. The introduction of this alpha value serves to normalize tests conducted with various ranks, since the magnitude of the weight updates is proportional to these ranks. By establishing a standard alpha value of 8, the authors are able to facilitate a fair comparison of LoRAs with different ranks, ensuring consistency in the size of the weight updates throughout training.

TL;DR: Quoting the paper,

> ...tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately.

## Todo

### Todo (ASAP)

- ~~Image Bucketing~~
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

- [Diffusers](https://huggingface.co/docs/diffusers/index)
- [Kohya Trainer](https://github.com/kohya-ss/sd-scripts)
- And [Stability AI](https://github.com/Stability-AI/generative-models)
