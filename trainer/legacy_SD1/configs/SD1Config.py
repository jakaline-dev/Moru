from dataclasses import dataclass, field
from typing import List, Optional

from configs.BaseConfig import BaseConfig, TrainerConfig


@dataclass
class LoRAConfig:
    r: int = 16
    target_modules: Optional[List[str]] = None
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    rank_pattern: Optional[dict] = field(default_factory=lambda: {})
    alpha_pattern: Optional[dict] = field(default_factory=lambda: {})


@dataclass
class PEFTConfig:
    type: str = "LoRA"
    lr: float = 0.0005
    parameters: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class SD1TrainerConfig(TrainerConfig):
    clip_skip: int = 1
    noise_offset: float = 0.0
    min_snr: int = 0


@dataclass
class TextualInversionConfig:
    placeholder_token: str = ""
    initializer_token: str = ""
    num_vectors: int = 1
    lr: float = 0.0005


@dataclass
class SD1Config(BaseConfig):
    trainer: SD1TrainerConfig = field(default_factory=SD1TrainerConfig)
    unet_peft: Optional[PEFTConfig] = field(
        default_factory=PEFTConfig(
            lr=0.0005,
            parameters=LoRAConfig(
                target_modules=[
                    "proj_in",
                    "proj_out",
                    "to_q",
                    "to_k",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                    "ff.net.2",
                ]
            ),
        )
    )
    text_encoder_peft: Optional[PEFTConfig] = field(
        default_factory=PEFTConfig(
            lr=0.00005,
            parameters=LoRAConfig(
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "out_proj",
                    "fc1",
                    "fc2",
                ]
            ),
        )
    )
    textual_inversion: Optional[TextualInversionConfig] = field(
        default_factory=TextualInversionConfig
    )
    is_train_unet: bool = False
    is_train_text_encoder: bool = False
