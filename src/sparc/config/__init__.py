"""Typed configuration: schema, loading, and dataset paths."""

from sparc.config.loader import load_config, save_config, to_dict
from sparc.config.schema import DownstreamConfig, PretrainConfig

__all__ = ["DownstreamConfig", "PretrainConfig", "load_config", "save_config", "to_dict"]
