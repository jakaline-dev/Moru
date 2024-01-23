from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Union

class Fabric(BaseModel):
    accelerator: str
    strategy: str
    devices: str
    precision: str

class MaxTrain(BaseModel):
    method: str
    value: int

class Trainer(BaseModel):
    grad_accum_steps: int
    use_xformers: bool
    cache_vae_outputs: bool
    cache_te_outputs: bool
    max_train: MaxTrain
    clip_skip: int
    noise_offset: float
    min_snr: int

class InitArgs(BaseModel):
    lr: float

class Optimizer(BaseModel):
    type: str
    lr: dict

class LRScheduler(BaseModel):
    name: str
    init_args: dict

class Preprocess(BaseModel):
    max_chunk: int
    min_chunk: int
    caption_template: List[str]

class Dataset(BaseModel):
    random_crop: bool
    random_flip: bool
    shuffle_tags: bool
    caption_dropout: float

class Dataloader(BaseModel):
    batch_size: int
    drop_last: bool
    pin_memory: bool
    num_workers: int
    persistent_workers: bool

class LoggingSave(BaseModel):
    every: str
    value: int

class LoggingSamplePipeline(BaseModel):
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_inference_steps: int

class LoggingSample(BaseModel):
    every: str
    value: int
    pipeline: LoggingSamplePipeline

class Logging(BaseModel):
    logger: Optional[str]
    save: LoggingSave
    sample: LoggingSample

class PEFTParameters(BaseModel):
    r: int
    target_modules: List[str]
    lora_alpha: int
    lora_dropout: float
    bias: str
    rank_pattern: dict
    alpha_pattern: dict

class PEFT(BaseModel):
    type: str
    lr: float
    parameters: PEFTParameters

class Config(BaseModel):
    model_type: str = "SDXL"
    train_lora: bool
    train_unet: bool
    train_text_encoder_1: bool
    train_text_encoder_2: bool
    seed: int
    name: str
    fabric: Fabric
    trainer: Trainer
    do_sample: bool = True
    sample_strategy: Literal['steps', 'epoch', 'no'] = 'steps'
    sample_steps: Optional[Union[int, float]] = 500
    save_strategy: Literal['steps', 'epoch', 'no'] = 'steps'
    save_steps: Optional[Union[int, float]] = 500
    gradient_accumulation_steps: int = 1
    optimizer: Optimizer
    lr: float = 1e-5
    lr_unet: Optional[float]
    lr_text_encoder_1: Optional[float]
    lr_text_encoder_2: Optional[float]
    lr_scheduler: LRScheduler
    cache_vae_outputs: bool
    cache_te_outputs: bool
    noise_offset: float
    min_snr: Union[int, float]
    preprocess: Preprocess
    dataset: Dataset
    dataloader: Dataloader
    logging: Logging
    unet_peft: List[PEFT]
    text_encoder_peft: List[PEFT]

# Example of how to create an instance of Config with your data
config_data = {
    # ... fill in with your configuration data ...
}

config = Config(**config_data)
