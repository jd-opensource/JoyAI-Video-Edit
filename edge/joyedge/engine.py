"""ctypes binding for the joyedge C++ engine (edge/engine, built with build.sh)."""
from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np


class ModelParams(C.Structure):
    _fields_ = [("n_threads", C.c_int32), ("stream_window", C.c_int32), ("verbose", C.c_int32)]


class Config(C.Structure):
    _fields_ = [
        ("hidden_size", C.c_int32), ("heads_num", C.c_int32), ("depth", C.c_int32),
        ("depth_in_file", C.c_int32), ("in_channels", C.c_int32), ("out_channels", C.c_int32),
        ("text_dim", C.c_int32), ("mlp_hidden", C.c_int32), ("rope_dims", C.c_int32 * 3),
        ("rope_theta", C.c_float), ("source_id_rope_dim", C.c_int32),
        ("source_id_rope_theta", C.c_float), ("scheduler_shift", C.c_float),
        ("num_train_timesteps", C.c_int32), ("n_params", C.c_uint64), ("file_bytes", C.c_uint64),
    ]


class SessionParams(C.Structure):
    _fields_ = [("lat_h", C.c_int32), ("lat_w", C.c_int32), ("n_steps", C.c_int32),
                ("max_temporal_ids", C.c_int32), ("shift", C.c_float), ("kv_f32", C.c_int32),
                ("txt_cache_interval", C.c_int32)]


class Timings(C.Structure):
    _fields_ = [("denoise_s", C.c_double), ("store_s", C.c_double), ("commit_s", C.c_double),
                ("total_s", C.c_double), ("kv_bytes", C.c_double), ("graph_bytes", C.c_double)]


def _find_library() -> str:
    env = os.environ.get("JOYEDGE_LIB")
    if env:
        return env
    ext = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
    here = Path(__file__).resolve().parents[1] / "engine" / "build"
    for cand in (here / f"libjoyedge.{ext}", here / f"joyedge.{ext}"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        f"libjoyedge.{ext} not found under {here}; run edge/engine/build.sh or set JOYEDGE_LIB."
    )


_lib = None


def lib():
    global _lib
    if _lib is None:
        L = C.CDLL(_find_library())
        fp = C.POINTER(C.c_float)
        L.joyedge_model_default_params.restype = ModelParams
        L.joyedge_session_default_params.restype = SessionParams
        L.joyedge_model_load.restype = C.c_void_p
        L.joyedge_model_load.argtypes = [C.c_char_p, ModelParams]
        L.joyedge_model_free.argtypes = [C.c_void_p]
        L.joyedge_model_get_config.argtypes = [C.c_void_p, C.POINTER(Config)]
        L.joyedge_session_new.restype = C.c_void_p
        L.joyedge_session_new.argtypes = [C.c_void_p, SessionParams]
        L.joyedge_session_free.argtypes = [C.c_void_p]
        L.joyedge_session_reset.argtypes = [C.c_void_p]
        L.joyedge_set_prompt.argtypes = [C.c_void_p, fp, C.c_int32]
        L.joyedge_set_reference_image.argtypes = [C.c_void_p, fp, C.c_int32, C.c_int32]
        L.joyedge_process_chunk.argtypes = [C.c_void_p, fp, fp, fp, C.c_int32]
        L.joyedge_chunk_index.argtypes = [C.c_void_p]
        L.joyedge_get_timings.argtypes = [C.c_void_p, C.POINTER(Timings)]
        L.joyedge_last_error.restype = C.c_char_p
        _lib = L
    return _lib


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(C.POINTER(C.c_float))


class EdgeDiT:
    """Streaming DiT on CPU.  One instance == one editing session."""

    def __init__(self, gguf_path: str, *, lat_h: int, lat_w: int, n_steps: int = 2,
                 max_temporal_ids: int = 2, n_threads: int = 0, stream_window: int = 0,
                 kv_f32: bool = False, txt_cache_interval: int = 0, verbose: bool = False):
        L = lib()
        mp = L.joyedge_model_default_params()
        mp.n_threads, mp.stream_window, mp.verbose = n_threads, stream_window, int(verbose)
        self._model = L.joyedge_model_load(str(gguf_path).encode(), mp)
        if not self._model:
            raise RuntimeError(f"joyedge: {L.joyedge_last_error().decode()}")
        self.config = Config()
        L.joyedge_model_get_config(self._model, C.byref(self.config))
        sp = L.joyedge_session_default_params()
        sp.lat_h, sp.lat_w, sp.n_steps = lat_h, lat_w, n_steps
        sp.max_temporal_ids, sp.kv_f32 = max_temporal_ids, int(kv_f32)
        sp.txt_cache_interval = int(txt_cache_interval)
        self._sess = L.joyedge_session_new(self._model, sp)
        if not self._sess:
            raise RuntimeError(f"joyedge: {L.joyedge_last_error().decode()}")
        self.lat_h, self.lat_w = lat_h, lat_w

    def _check(self, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"joyedge: {lib().joyedge_last_error().decode()}")

    def set_prompt(self, embeds) -> None:
        """embeds: [n_tokens, text_dim] hidden states of the condition encoder."""
        e = _f32(embeds)
        assert e.ndim == 2 and e.shape[1] == self.config.text_dim, e.shape
        self._check(lib().joyedge_set_prompt(self._sess, _ptr(e), e.shape[0]))

    def set_reference_image(self, latent) -> None:
        """latent: [C, h, w] normalized VAE latent of the reference image."""
        x = _f32(latent)
        assert x.ndim == 3 and x.shape[0] == self.config.in_channels, x.shape
        self._check(lib().joyedge_set_reference_image(self._sess, _ptr(x), x.shape[1], x.shape[2]))

    def process_chunk(self, noise, src_latent, *, update_history: bool = True) -> np.ndarray:
        """noise / src_latent: [C, lat_h, lat_w].  Returns the edited latent, same shape."""
        shape = (self.config.in_channels, self.lat_h, self.lat_w)
        n, s = _f32(noise).reshape(shape), _f32(src_latent).reshape(shape)
        out = np.empty(shape, dtype=np.float32)
        self._check(lib().joyedge_process_chunk(self._sess, _ptr(n), _ptr(s), _ptr(out), int(update_history)))
        return out

    def reset(self) -> None:
        lib().joyedge_session_reset(self._sess)

    @property
    def timings(self) -> dict:
        t = Timings()
        lib().joyedge_get_timings(self._sess, C.byref(t))
        return {k: getattr(t, k) for k, _ in Timings._fields_}

    def close(self) -> None:
        L = lib()
        if getattr(self, "_sess", None):
            L.joyedge_session_free(self._sess)
            self._sess = None
        if getattr(self, "_model", None):
            L.joyedge_model_free(self._model)
            self._model = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
