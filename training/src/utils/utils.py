import importlib


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def build_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])
    params = dict(config.get("params", {}))
    params.update(kwargs)

    pretrained_path = config.get("pretrained", None)
    if pretrained_path is not None and hasattr(cls, "from_pretrained"):
        return cls.from_pretrained(pretrained_path, **params)

    obj = cls(**params)
    return obj
