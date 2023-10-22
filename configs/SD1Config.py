from dataclasses import dataclass, field, replace
from typing import Dict, Union, Optional, List
from omegaconf import MISSING
from enum import StrEnum
from configs.BaseConfig import BaseConfig


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
class ModuleConfig:
    train: bool = True
    peft: PEFTConfig = field(default_factory=PEFTConfig)


@dataclass
class UNETConfig(ModuleConfig):
    noise_offset: float = 0.0
    snr_gamma: int = 0

    def __post_init__(self):
        if self.peft.parameters.target_modules is None:
            self.peft.parameters.target_modules = [
                "proj_in",
                "proj_out",
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ]


@dataclass
class CLIPConfig(ModuleConfig):
    clip_skip: int = 1

    def __post_init__(self):
        if self.peft.parameters.target_modules is None:
            self.peft.parameters.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "fc1",
                "fc2",
            ]


@dataclass
class SD1Config(BaseConfig):
    unet: UNETConfig = field(default_factory=UNETConfig)
    text_encoder: CLIPConfig = field(default_factory=CLIPConfig)
