# Moru Trainer

Diffusers meets Lightning Fabric

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

## Todo

[x] Image Bucketing\
[] Kohya LoRA format support\
[] Easy Installers\
[] Textual Inversion\
[] SDXL Finetuning\
[] SDXL LoRA Training

## Future Endeavours

- Image Padding / Masking
- New PEFT (LoRA-FA, iA3, Bit-fit, VeRA, ...)
- P+ (TI embeds with layers)
- NeTI (TI embeds with timesteps)
- Würstchen v3 (When it's up)

## Shoutouts

This trainer was built from the shoulders of many giants. Also, just giving a shoutout who led me to this rabbit hole.

- [Diffusers](https://huggingface.co/docs/diffusers/index)
- [Kohya Trainer](https://github.com/kohya-ss/sd-scripts)
- https://github.com/cloneofsimo/lora
- [Lightning Fabric](https://lightning.ai)
- And [Stability AI](https://github.com/Stability-AI/generative-models)
