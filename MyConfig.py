from dataclasses import dataclass, field
from typing import Dict, Union
from functools import partial
from omegaconf import MISSING
from enum import StrEnum


class TrainType(StrEnum):
    LoRA = "LoRA"
    Finetune = "Finetune"
    TI = "TI"


class ModelType(StrEnum):
    SD1 = "SD1"
    SDXL = "SDXL"


class PrecisionType(StrEnum):
    bf16_mixed = "bf16-mixed"
    bf16_true = "bf16-true"
    fp16_mixed = "fp16-mixed"


class StepOrEpoch(StrEnum):
    step = "step"
    epoch = "epoch"


@dataclass
class PretrainedModelConfig:
    checkpoint_path: str = MISSING
    type: ModelType = "SD1"
    clip_skip: int = 1


@dataclass
class MaxTrainConfig:
    method: StepOrEpoch = "epoch"
    value: int = 10


@dataclass
class SaveConfig:
    every: StepOrEpoch | None = "epoch"
    value: int = 1


@dataclass
class SamplePipelineConfig:
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20


@dataclass
class SampleConfig:
    every: StepOrEpoch | None = "epoch"
    value: int = 1
    num_images: int = 1
    pipeline: SamplePipelineConfig = field(default_factory=SamplePipelineConfig)


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: str = "auto"
    precision: PrecisionType = "bf16_true"
    batch_size: int = 1
    grad_accum_steps: int = 1
    use_xformers: bool = False
    max_train: MaxTrainConfig = field(default_factory=MaxTrainConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


@dataclass
class Optimizer:
    name: str = "AdamW"
    init_args: Dict[str, Union[str, int, float]] = field(default_factory=lambda: {"lr": 0.0001})

@dataclass
class LRScheduler:
    name: str = "cosine_with_restarts"
    init_args: Dict[str, Union[str, int, float]] = field(default_factory=lambda: {"num_cycles": 3, "num_warmup_steps": 50})

@dataclass
class LoraUnitConfig:
    enable_train: bool = True
    rank: int = 32
    network_alpha: int = 32
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)


@dataclass
class ModelConfig:
    type: ModelType = "SD1"
    clip_skip: int = 2
    noise_offset: float = 0.0
    snr_gamma: int = 0
    lora_unet: LoraUnitConfig = field(default_factory=LoraUnitConfig)
    lora_te: LoraUnitConfig = field(default_factory=LoraUnitConfig)


@dataclass
class PreprocessConfig:
    folder_path: str = MISSING
    max_chunk: int = 64
    min_chunk: int = 32
    cache_vae_outputs: bool = False
    cache_te_outputs: bool = False


@dataclass
class DatasetConfig:
    random_crop: bool = True
    random_flip: bool = True
    shuffle_tags: bool = False


@dataclass
class DataloaderConfig:
    batch_size: int = 1
    drop_last: bool = False
    pin_memory: bool = True
    num_workers: int = 1
    persistent_workers: bool = True


@dataclass
class BucketConfig:
    enable: bool = True


@dataclass
class MyConfig:
    seed: int | None = 42
    name: str = MISSING
    type: TrainType = "LoRA"
    pretrained_model: PretrainedModelConfig = field(
        default_factory=PretrainedModelConfig
    )
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    bucket: BucketConfig = field(default_factory=BucketConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
