"""joyedge - CPU / edge runtime tooling for JoyAI-Video-Edit.

* ``export_gguf``  - convert a DiT checkpoint to a (quantized) GGUF file, streaming.
* ``engine``       - ctypes binding of the ggml-based C++ engine (``libjoyedge``).
* ``reference``    - pure-PyTorch streaming rollout used as the numerical reference.
* ``budget``       - compute / latency model for sizing a configuration to a device.
"""
