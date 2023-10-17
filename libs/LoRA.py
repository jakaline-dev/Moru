import torch
from diffusers.models.lora import LoRALinearLayer
from diffusers.loaders import PatchedLoraProjection


def set_unet_lora_linear_layer(attn_module_attribute, rank, network_alpha):
    attn_module_attribute.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module_attribute.in_features,
            out_features=attn_module_attribute.out_features,
            rank=rank,
            network_alpha=network_alpha,
        )
    )


def set_te_lora_linear_layer(model, rank, network_alpha):
    model = PatchedLoraProjection(
        model,
        rank=rank,
        network_alpha=network_alpha,
    )
    return model


def unet_lora_state_dict(unet, alpha):
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "lora_layer"):
            lora_layer = getattr(module, "lora_layer")
        else:
            continue

        if lora_layer is not None:
            current_lora_layer_sd = lora_layer.state_dict()
            lora_state_dict[f"lora_unet_{name.replace('.', '_')}.alpha"] = torch.tensor(
                alpha, dtype=unet.dtype
            )
            for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                lora_state_dict[
                    f"lora_unet_{name.replace('.', '_')}.lora_{lora_layer_matrix_name}"
                ] = lora_param

    return lora_state_dict


def text_encoder_lora_state_dict(text_encoder, alpha):
    lora_state_dict = {}

    for name, module in text_encoder.named_modules():
        if hasattr(module, "lora_linear_layer"):
            lora_layer = getattr(module, "lora_linear_layer")
            if lora_layer is not None:
                current_lora_layer_sd = lora_layer.state_dict()
                lora_state_dict[
                    f"lora_te_{name.replace('.', '_')}.alpha"
                ] = torch.tensor(alpha, dtype=text_encoder.dtype)
                for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                    lora_state_dict[
                        f"lora_te_{name.replace('.', '_')}.lora_{lora_layer_matrix_name}"
                    ] = lora_param
    return lora_state_dict
