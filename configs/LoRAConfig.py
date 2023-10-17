from .BaseConfig import BaseConfig
from dataclasses import dataclass, field
from typing import List


@dataclass
class PEFTLoRAConfig:
    r: int = 16
    target_modules: List[str] = field(default_factory=lambda: [])
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"


@dataclass
class LoRAConfig(BaseConfig):
    lora_unet: List[PEFTLoRAConfig] = field(default_factory=lambda: [PEFTLoRAConfig])
    lora_te: List[PEFTLoRAConfig] = field(default_factory=lambda: [PEFTLoRAConfig])
