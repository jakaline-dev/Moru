from dataclasses import dataclass, field
from typing import Dict, Union, Optional, List
from omegaconf import MISSING
from enum import StrEnum


class TrainType(StrEnum):
    LoRA = "LoRA"
    Finetune = "Finetune"
    TI = "TI"


class StepOrEpoch(StrEnum):
    step = "step"
    epoch = "epoch"
    skip = "skip"


@dataclass
class PretrainedModelConfig:
    checkpoint_path: str = MISSING
    architecture: str = "SD1"
    clip_skip: int = 1


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


@dataclass
class SamplePipelineConfig:
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    clip_skip: int = "${pretrained_model.clip_skip}"


@dataclass
class SampleConfig:
    every: StepOrEpoch = "epoch"
    value: int = 1
    pipeline: SamplePipelineConfig = field(default_factory=SamplePipelineConfig)


@dataclass
class TrainerConfig:
    clip_skip: int = "${pretrained_model.clip_skip}"
    batch_size: int = 1
    grad_accum_steps: int = 1
    use_xformers: bool = False
    noise_offset: float = 0.0
    snr_gamma: int = 0
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
    max_chunk: int = 64
    min_chunk: int = 32
    cache_vae_outputs: bool = False
    cache_te_outputs: bool = False


@dataclass
class DatasetConfig:
    random_crop: bool = True
    random_flip: bool = True
    shuffle_tags: bool = False
    caption_dropout: float = 0.0


@dataclass
class DataloaderConfig:
    batch_size: int = "${trainer.batch_size}"
    drop_last: bool = True
    pin_memory: bool = True
    num_workers: int = 1
    persistent_workers: bool = True


@dataclass
class DataPipelineConfig:
    path: Optional[str] = MISSING
    hf_repo: Optional[str] = MISSING
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)


@dataclass
class LoggingConfig:
    logger: Optional[str] = None
    save: SaveConfig = field(default_factory=SaveConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


@dataclass
class BaseConfig:
    seed: int | None = 42
    name: str = MISSING
    run_name: str = MISSING
    type: TrainType = "LoRA"
    fabric: FabricConfig = field(default_factory=FabricConfig)
    pretrained_model: PretrainedModelConfig = field(
        default_factory=PretrainedModelConfig
    )
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConig = field(default_factory=OptimizerConig)
    lr_scheduler: Union[LRSchedulerConfig, None] = field(
        default_factory=LRSchedulerConfig
    )
    datapipe: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
