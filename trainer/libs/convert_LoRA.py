# SOURCE: https://github.com/huggingface/peft/blob/main/examples/lora_dreambooth/convert_peft_sd_lora_to_kohya_ss.py

from typing import Dict

import torch

# Default kohya_ss LoRA replacement modules
# https://github.com/kohya-ss/sd-scripts/blob/c924c47f374ac1b6e33e71f82948eb1853e2243f/networks/lora.py#L664
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_ADAPTER_NAME = "default"


def get_module_kohya_state_dict(
    weights, prefix: str, network_alpha: int
) -> Dict[str, torch.Tensor]:
    kohya_ss_state_dict = {}
    for peft_key, weight in weights.items():
        kohya_key = prefix + "_" + peft_key
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(network_alpha).to(
                dtype=weight.dtype
            )

    return kohya_ss_state_dict
