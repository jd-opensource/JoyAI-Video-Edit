import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def build_from_config(config, **kwargs):
    if "target" not in config:
        if config in ("__is_first_stage__", "__is_unconditional__"):
            return None
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])
    params = dict(config.get("params", {}))
    params.update(kwargs)

    pretrained_path = config.get("pretrained", None)
    if pretrained_path is not None and hasattr(cls, "from_pretrained"):
        return cls.from_pretrained(pretrained_path, **params)

    obj = cls(**params)
    return obj
