import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline


@torch.inference_mode()
def sample(
    name,
    config,
    device=None,
    unet=None,
    text_encoder=None,
    text_encoder_2=None,
    vae=None,
    tokenizer=None,
    tokenizer_2=None,
    noise_scheduler_config=None,
):
    # noise_scheduler = EulerDiscreteScheduler.from_config(
    #     noise_scheduler_config, timestep_type="linspace", rescale_betas_zero_snr=False, sigma_min=None, sigma_max=None
    # )
    noise_scheduler = DPMSolverMultistepScheduler.from_config(
        noise_scheduler_config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
    )

    pipeline = StableDiffusionXLPipeline(
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        unet=unet,
    )

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        torch.Generator(device=device).manual_seed(config.seed) if config.seed else None
    )

    images = [
        pipeline(**config.sample_pipeline.model_dump(), generator=generator).images[0]
        for _ in range(config.num_sample_images)
    ]

    # for tracker in accelerator.trackers:
    #     if tracker.name == "tensorboard":
    #         np_images = np.stack([np.asarray(img) for img in images])
    #         tracker.writer.add_images(
    #             "validation", np_images, epoch, dataformats="NHWC"
    #         )
    #     if tracker.name == "wandb":
    #         tracker.log(
    #             {
    #                 "validation": [
    #                     wandb.Image(
    #                         image,
    #                         caption=f"{i}: {config.sample_pipeline.prompt}",
    #                     )
    #                     for i, image in enumerate(images)
    #                 ]
    #             }
    #         )

    for i, image in enumerate(images):
        if config.num_sample_images > 1:
            image.save(f"{name}_{i}.png")
        else:
            image.save(f"{name}.png")
    del images
    del pipeline
    torch.cuda.empty_cache()
