"""Miscellaneous utility functions."""


import yaml
from typing import Dict


def load_yaml(yaml_file: str) -> Dict:
    """Loads contents from yaml file.

    Args:
        yaml_file (str): Path of yaml file.
    
    Returns:
        contents (dict): Contents of yaml file.
    """

    # load contents from yaml file:
    with open(yaml_file, "r") as yaml_file:
        contents = yaml.safe_load(yaml_file)
    
    return contents

