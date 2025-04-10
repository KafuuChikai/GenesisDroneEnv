"""
This module contains functions to process the configuration for the train and evaluate scripts.
"""

import argparse, os, yaml
from typing import Union
from collections.abc import Mapping
from datetime import datetime


def process_config(
    args: argparse.Namespace,
    current_dir: Union[str, os.PathLike],
    envkey: dict,
    trainkey: dict,
) -> dict:

    # read the config file
    raw_env_cfg = _get_config(envkey[args.exp_name], current_dir, "envs")
    train_cfg = _get_config(trainkey[args.exp_name], current_dir, "train")

    # update the config_dict with envconfig, algconfig, and runconfig
    raw_env_cfg, train_cfg = _read_all_config(args, current_dir, raw_env_cfg, train_cfg)

    env_cfg = raw_env_cfg.get("env_cfg", {})
    obs_cfg = raw_env_cfg.get("obs_cfg", {})
    reward_cfg = raw_env_cfg.get("reward_cfg", {})
    command_cfg = raw_env_cfg.get("command_cfg", {})

    combined_config = {
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "train_cfg": train_cfg,
    }

    _save_config(args, combined_config, current_dir)

    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def _save_config(args, config_dict: dict, current_dir: Union[str, os.PathLike]) -> None:
    """Save the config dictionary to a YAML file.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    config_save_path = os.path.join(
        current_dir,
        "logs",
        f"{args.exp_name}_{timestamp}",
        "config.yaml",
    )
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    
    with open(config_save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def _get_config(config_name: str, current_dir: Union[str, os.PathLike], subfolder: Union[str, os.PathLike]) -> dict:
    """Get the config dictionary from the YAML file.

    Parameters
    ----------
    config_name : str
        Name of the config file (without extension).
    current_dir : Union[str, os.PathLike]
        Current directory of the script.
    subfolder : Union[str, os.PathLike]
        Subfolder where the config file is located.

    Returns
    -------
    config_dict : dict
        Dictionary containing the configuration parameters.

    """
    with open(os.path.join(current_dir, "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict


def _read_all_config(
    args: argparse.Namespace, current_dir: Union[str, os.PathLike], env_cfg: dict, train_cfg: dict,
) -> dict:
    """Read the default config file.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.
    envconfig : dict
        Environment configuration dictionary.
    algconfig : dict
        Algorithm configuration dictionary.
    runconfig : dict
        Runner configuration dictionary.

    Returns
    -------
    config_dict : dict
        Dictionary containing the configuration parameters.

    """
    # Update the train_cfg with command line arguments
    exp_name = getattr(args, "exp_name", None)
    if exp_name is not None:
        train_cfg["experiment_name"] = exp_name

    max_steps = getattr(args, "max-iterations", None)
    if max_steps is not None:
        train_cfg["max-iterations"] = max_steps
    
    return env_cfg, train_cfg


def get_train_cfg(exp_name, max_iterations, current_dir=None, config_name="train_config", subfolder="train"):
    
    train_cfg_dict = _get_config(config_name, current_dir, subfolder)

    return train_cfg_dict


def get_cfgs(current_dir=None, config_name="env_config", subfolder="envs"):
    
    config = _get_config(config_name, current_dir, subfolder)

    # read the config file
    env_cfg = config.get("env_cfg", {})
    obs_cfg = config.get("obs_cfg", {})
    reward_cfg = config.get("reward_cfg", {})
    command_cfg = config.get("command_cfg", {})

    return env_cfg, obs_cfg, reward_cfg, command_cfg



# -------------------utils-------------------
def recursive_dict_update(d: dict, u: dict) -> dict:
    """Recursively update a dictionary with another dictionary.

    Parameters
    ----------
    d : dict
        The original dictionary to be updated.
    u : dict
        The dictionary with new values to update the original dictionary.

    Returns
    -------
    d : dict
        The updated dictionary.

    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            if not isinstance(d.get(k), Mapping):
                d[k] = {}
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def recursive_enum_mapping(d: dict, m: dict) -> dict:
    """Recursively map enum values in the dictionary.

    Parameters
    ----------
    d : dict
        The original dictionary to be updated.
    m : dict
        The dictionary with enum mappings.

    Returns
    -------
    d : dict
        The updated dictionary with enum values.

    """
    for k, v in d.items():
        if isinstance(v, Mapping):
            d[k] = recursive_enum_mapping(d.get(k, {}), m)
        elif k in m:
            d[k] = getattr(m[k], v, None)
            if d[k] is None:
                raise ValueError(f"Invalid value '{v}' for key '{k}'.")
    return d

def fix_none_values(d: dict) -> None:
    """Recursively fix 'None' string values in the dictionary.

    Parameters
    ----------
    d : dict
        The original dictionary to be updated.

    """
    for k, v in d.items():
        if isinstance(v, dict):
            fix_none_values(v)
        elif v == "None":
            d[k] = None