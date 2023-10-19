# Moru Trainer

Diffusers meets Lightning Fabric.

2x faster than Kohya trainer.

Not production-ready - not ready at all atm.
(Tested in python 3.11, pytorch 2.1.0, CUDA 12.1)

## Features

- SD1 LoRA Training

## Usage

Currently only SD1 LoRA training available.
Currently only for Windows - will also support WSL and Linux soon.

1. git pull --autostash
2. Open install_windows.bat and let it install
3. Fill in the name, base.checkpoint_path, preprocess.folder_path in the config-example.yaml file.
4. Open cmd_windows.bat
5. Type this:

```bash
python run.py --config config-example.yaml
```

## Remarks

### Remarks about precision

There are two factors:

- A: The precision of the original model (Related to vram usage)
- B: The precision of computations (Related to training speed)

BF16-mixed: FP32, BF16
BF16-true: BF16, BF16 (Half the VRAM)

When not finetuning (LoRA, TI), always go with bf16-true.
If you are finetuning, you have two options: bf16-true or bf16-mixed. If you go with bf16-true, the output of the finedtuned model will be set to bf16 dtype, which you cannot retrain with fp32 or fp16 ever again. Training with bf16-true will also have lower accuracy than with fp32 training, ~~but the drop is almost negligible, as SD has a lot of parameters that can make up for it.~~ Use AnyPrecisionAdamW optimizer with bf16-true for maximum accuracy, and half the vram!

No fp16 - fp16 causes numerical instability (your loss turning into none). Technically this can be alleviated by dynamically controlling the scaling factor of loss when underflow hits, but it's really not that worth it to implement this.

### Remarks about difference of LoRA training with Kohya's and Moru Trainer

As of today, kohya's trainer trains these layers when training LoRA.

- text encoder attention modules: q, k, v, o
- text encoder MLP modules: ff1, ff2
- unet projection : proj_in, proj_out
- unet attention modules: q, k, v, o
- unet MLP modules: ff.net.0, ff.net.2

However, Moru trainer does not train proj_in and proj_out of unet, as this is a convolution layer in SD1.

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
- [Lightning Fabric](https://lightning.ai)
- And [Stability AI](https://github.com/Stability-AI/generative-models)
