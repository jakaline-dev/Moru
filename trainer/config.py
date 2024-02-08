from pydantic import BaseModel
from typing import Optional, List, Literal, Union


class Fabric(BaseModel):
    accelerator: str
    strategy: str
    devices: str
    precision: str


class Accelerator(BaseModel):
    mixed_precision: str
    report_to: Optional[Literal["all", "tensorboard", "wandb"]]


class AdamOptimizer(BaseModel):
    type: Literal["AdamW", "AdamW8bit", "AnyPrecisionAdamW"]
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    eps: float = 1e-08


class LRScheduler(BaseModel):
    type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
        "piecewise_constant",
    ] = "constant"
    num_warmup_steps: Optional[int] = None


class Dataset(BaseModel):
    local_path: str
    image_column: str = "image"
    text_column: str = "text"

    random_crop: bool
    random_flip: bool
    # shuffle_tags: bool
    # caption_dropout: float
    # max_chunk: int
    # min_chunk: int
    # caption_template: List[str]


class Dataloader(BaseModel):
    batch_size: int = 1
    drop_last: bool = False
    pin_memory: bool = False
    num_workers: int = 0
    persistent_workers: bool = False


class SamplePipeline(BaseModel):
    prompt: str
    negative_prompt: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int


class LoraConfig_(BaseModel):
    r: int
    target_modules: List[str]
    lora_alpha: int
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False
    init_lora_weights: Union[bool, Literal["gaussian", "loftq"]] = "gaussian"
    # rank_pattern: dict
    # alpha_pattern: dict


# ["to_k", "to_q", "to_v", "to_out.0"]
# ["q_proj", "k_proj", "v_proj", "out_proj"]


class Config(BaseModel):
    name: str

    checkpoint_path: str
    vae_path: Optional[str] = None

    # fabric: Fabric
    accelerator: Accelerator
    seed: int

    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    xformers: bool = False

    max_train_steps: Optional[int] = None
    num_train_epochs: Optional[int] = 10

    sample_strategy: Literal["steps", "epoch", "no"] = "steps"
    sample_steps: Optional[Union[int, float]] = 100
    sample_pipeline: SamplePipeline
    num_sample_images: int = 1

    save_strategy: Literal["steps", "epoch", "only_last"] = "steps"
    save_steps: Optional[Union[int, float]] = 200

    optimizer: AdamOptimizer
    lr_scheduler: LRScheduler
    dataset: Dataset
    dataloader: Dataloader

    noise_offset: float = None
    min_snr: Union[int, float] = None

    max_grad_norm: float = 1.0

    lr_unet: Optional[float] = 0.0
    lr_text_encoder: Optional[float] = 0.0
    lr_text_encoder_2: Optional[float] = 0.0

    cache_vae: bool = False
    cache_text_encoder: bool = False
    cache_text_encoder_2: bool = False

    peft_unet: LoraConfig_
    peft_text_encoder: LoraConfig_
    peft_text_encoder_2: LoraConfig_
