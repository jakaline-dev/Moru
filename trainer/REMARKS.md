# Remarks

## Remarks about precision

There are two factors:

- A: The precision of the original model (Related to vram usage)
- B: The precision of computations (Related to training speed)

BF16-mixed: A: FP32, B: BF16\
BF16-true: A: BF16, B: BF16 (Half the VRAM)

When not finetuning (LoRA, TI), always go with bf16-true.

If you are finetuning, you have two options: bf16-true or bf16-mixed.

If you go with bf16-true, the output of the finedtuned model will be set to bf16 dtype, which you cannot retrain with fp32 or fp16 ever again.

Also, when using bf16-true, use AnyPrecisionAdamW optimizer, as it handles the accuracy drop of bf16 computations.

Stay away from fp16-mixed. If your GPU card is not an NVIDIA card above 3000x series, try using colab / cloud instances.

## Remarks about LoRA trainable layers

As of today, kohya's trainer trains these layers when training LoRA.

- text encoder attention modules: q, k, v, o
- text encoder MLP modules: ff1, ff2
- unet projection : proj_in, proj_out
- unet attention modules: q, k, v, o
- unet MLP modules: ff.net.0, ff.net.2

While those are the most important layers of the diffusion model that's worth training, further test is needed to figure out which layers are the most effective.

There is no golden rule of thumb - but you can start from training the attention modules of unet, and gradually add some layers to analyze ths differences.

```yaml
unet:
  type: LoRA
  parameters:
    target_modules:
      - to_q
      - to_k
      - to_v
      - to_out.0
    bias: "none"
te: null
```

### Remarks about 'network_alpha' value

In the LoRA paper, the authors scale the weight updates by a factor of network_alpha divided by the LoRA rank. The introduction of this alpha value serves to normalize tests conducted with various ranks, since the magnitude of the weight updates is proportional to these ranks. By establishing a standard alpha value of 8, the authors are able to facilitate a fair comparison of LoRAs with different ranks, ensuring consistency in the size of the weight updates throughout training.

TL;DR: Quoting the paper,

> ...tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately.
