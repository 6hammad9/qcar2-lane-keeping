import yaml
from types import SimpleNamespace


def dict_to_namespace(d):

    if isinstance(d, dict):
        return SimpleNamespace(
            **{
                k: dict_to_namespace(v)
                for k, v in d.items()
            }
        )

    return d


def load_config(path):

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return dict_to_namespace(data)