from .BaseConfig import BaseConfig
from dataclasses import dataclass, field
from typing import List, Union, Optional


@dataclass
class LoRAParametersConfig:
    r: int = 16
    target_modules: Optional[List[str]] = None
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    rank_pattern: Optional[dict] = field(default_factory=lambda: {})
    alpha_pattern: Optional[dict] = field(default_factory=lambda: {})


@dataclass
class LoRAModuleConfig:
    type: str = "LoRA"
    lr: float = 0.0005
    parameters: LoRAParametersConfig = field(default_factory=LoRAParametersConfig)


@dataclass
class PEFTInnerConfig:
    unet: Optional[LoRAModuleConfig] = field(default_factory=LoRAModuleConfig)
    te: Optional[LoRAModuleConfig] = field(default_factory=LoRAModuleConfig)


@dataclass
class PEFTConfig(BaseConfig):
    peft: PEFTInnerConfig = field(default_factory=PEFTInnerConfig)
