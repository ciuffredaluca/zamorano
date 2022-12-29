from omegaconf import OmegaConf, DictConfig
import pkg_resources

from typing import *


def load_config() -> DictConfig:
    config_path = pkg_resources.resource_filename(__name__, "config.yaml")
    return OmegaConf.load(config_path)
