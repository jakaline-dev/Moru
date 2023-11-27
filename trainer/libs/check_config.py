from datetime import datetime


def check_config(config):
    if config.text_encoder_peft and config.textual_inversion:
        raise Exception(
            "Text encoder LoRA training + Textual Inversion Training is not possible"
        )

    if config.text_encoder_peft or config.textual_inversion:
        config.is_train_text_encoder = True
    else:
        config.is_train_text_encoder = False

    if config.unet_peft:
        config.is_train_unet = True

    config.run_name = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{config.name}"

    return config
