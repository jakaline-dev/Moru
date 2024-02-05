from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List, Optional, Union

from omegaconf import MISSING


class StepOrEpoch(StrEnum):
    step = "step"
    epoch = "epoch"
    skip = "skip"


@dataclass
class PathConfig:
    base_checkpoint_path: str = MISSING
    train_data_path: str = MISSING
    validation_path: str = MISSING


@dataclass
class FabricConfig:
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: str = "auto"
    precision: str = "bf16-true"


@dataclass
class MaxTrainConfig:
    method: StepOrEpoch = "epoch"
    value: int = 10


@dataclass
class SaveConfig:
    every: StepOrEpoch = "epoch"
    value: int = 1
    output_diffusers: bool = False
    output_kohya_ss: bool = True


@dataclass
class SamplePipelineConfig:
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20


@dataclass
class SampleConfig:
    every: StepOrEpoch = "epoch"
    value: int = 1
    pipeline: SamplePipelineConfig = field(default_factory=SamplePipelineConfig)


@dataclass
class TrainerConfig:
    grad_accum_steps: int = 1
    gradient_checkpointing: bool = True
    use_xformers: bool = False
    cache_vae_outputs: bool = False
    cache_te_outputs: bool = False
    max_train: MaxTrainConfig = field(default_factory=MaxTrainConfig)


@dataclass
class OptimizerConig:
    name: str = "AdamW"
    init_args: Dict[str, Union[str, int, float]] = field(
        default_factory=lambda: {"lr": 0.0001}
    )


@dataclass
class LRSchedulerConfig:
    name: str = "cosine_with_restarts"
    init_args: Dict[str, Union[str, int, float]] = field(
        default_factory=lambda: {"num_cycles": 3, "num_warmup_steps": 50}
    )


@dataclass
class PreprocessConfig:
    max_chunk: int = 128
    min_chunk: int = 32
    caption_template: Optional[List[str]] = field(default_factory=lambda: [])


@dataclass
class DatasetConfig:
    random_crop: bool = True
    random_flip: bool = True
    shuffle_tags: bool = False
    caption_dropout: float = 0.0


@dataclass
class DataloaderConfig:
    batch_size: int = 1
    drop_last: bool = True
    pin_memory: bool = True
    num_workers: int = 1
    persistent_workers: bool = True


@dataclass
class DataPipelineConfig:
    path: Optional[str] = MISSING
    hf_repo: Optional[str] = MISSING


@dataclass
class LoggingConfig:
    logger: Optional[str] = None
    save: SaveConfig = field(default_factory=SaveConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


@dataclass
class BaseConfig:
    seed: Union[int, None] = 42
    name: str = MISSING
    run_name: str = MISSING
    fabric: FabricConfig = field(default_factory=FabricConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConig = field(default_factory=OptimizerConig)
    lr_scheduler: Union[LRSchedulerConfig, None] = field(
        default_factory=LRSchedulerConfig
    )
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
