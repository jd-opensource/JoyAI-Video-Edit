import os
import sys

import loguru

_logger = None


class NullLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def bind(self, **kwargs):
        return self


def setup_logger(exp_dir: str | None = None):
    global _logger

    loguru.logger.remove()
    if int(os.getenv("RANK", 0)) > 0:
        _logger = NullLogger()
        return _logger

    _logger = loguru.logger
    options = dict(
        level="INFO",
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    _logger.add(sys.stderr, **options)
    if exp_dir is not None:
        _logger.add(os.path.join(exp_dir, "train.log"), colorize=False, encoding="utf-8", **options)
    return _logger


def get_logger():
    return _logger if _logger is not None else setup_logger()


__all__ = ["setup_logger", "get_logger"]
