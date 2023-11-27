import argparse
import sys

import torch
from configs.SD1Config import SD1Config
from libs.check_config import check_config
from omegaconf import OmegaConf
from train_SD1 import main as train_SD1

if __name__ == "__main__":
    # set environment
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(torch.device("cuda"))
        if capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.allow_tf32 = True
            print("TF32 Enabled")
            torch.set_float32_matmul_precision("medium")

    # argparse
    parser = argparse.ArgumentParser(description="Provide configuration file.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        type=str,
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--debug-config",
        action="store_true",
        help="Save a default configuration YAML file",
    )
    args = parser.parse_args()

    # omegaconf
    config = OmegaConf.structured(SD1Config)
    if args.debug_config:
        print("Write debug config")
        with open("config_.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(config))
        sys.exit(0)

    yaml_config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, yaml_config)
    config = check_config(config)
    train_SD1(config)
