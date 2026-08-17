"""Fix torch inductor: warm restarts re-run coordinate-descent autotuning
(GPU re-benchmarking, seconds per kernel) even though every compile artifact
is already cached on disk.

Defects (verified against each release branch of
`torch/_inductor/runtime/{autotune_cache,triton_heuristics}.py`), all hit
with ``torch.compile(mode="max-autotune*")``:

1. [2.9 - at least 2.13] ``.best_config`` files are keyed by kernel source
   hash only, but validated against the *caller's* candidate-config-set hash
   (``configs_hash``). Two graphs that contain the same kernel source with
   different candidate sets (e.g. a Triton-bundle restore that only kept the
   winner vs. a fresh compile enumerating full heuristics) permanently
   invalidate each other's entry: every process start re-runs coordinate
   descent for those kernels and rewrites the file (ping-pong), so the cache
   never converges.

2. [2.9 - 2.11 only; fixed upstream in 2.12 via pytorch#173391, issue
   pytorch#172819] ``CachingAutotuner.recheck_autotune_cache()``
   (bundle-restore path) only adopts a cached best config when
   ``len(configs) > 1``, which is false for bundles that restored exactly
   one compile result.

3. [2.9 - 2.11 only; same upstream fix] Even when adoption happens, the flag
   ``found_by_coordesc`` is not propagated to the adopted candidate
   ``Config``, and ``CachingAutotuner.run()`` re-runs coordinate descent
   whenever that flag is False.

Fixes installed here, gated per defect:

- ``AutotuneCache.read_best`` (defect 1, installed on 2.9-2.13): when the
  cached entry was produced by coordinate descent and coordesc is enabled,
  validate against the entry's own ``configs_hash`` instead of the caller's.
  Safe because the cache filename already pins the kernel source hash and
  ``torch_key()``, and the kernel source embeds the size hints — the tuned
  config is valid for this kernel regardless of which candidate list the
  caller enumerated.

- ``CachingAutotuner.recheck_autotune_cache`` (defects 2+3, installed on
  2.9-2.11 only): drop the ``len(configs) > 1`` requirement and propagate
  ``found_by_coordesc`` onto the adopted config so launch-time coordesc is
  skipped. On 2.12+ the upstream version of this method is kept as-is.

Set ``JOYOMNI_NO_COORDESC_CACHE_FIX=1`` to disable everything.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False

# Versions verified to carry each defect AND to expose the exact internal
# APIs this module patches (read_best/_load_cached_autotuning signatures,
# check_autotune_cache, found_by_coordesc flag protocol).
_READ_BEST_VERSIONS = ("2.9.", "2.10.", "2.11.", "2.12.", "2.13.")
_RECHECK_VERSIONS = ("2.9.", "2.10.", "2.11.")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("JOYOMNI_NO_COORDESC_CACHE_FIX", "").lower() in {"1", "true", "yes", "on"}:
        logger.warning("[inductor_autotune_fix] disabled via JOYOMNI_NO_COORDESC_CACHE_FIX")
        return
    try:
        import torch
        from torch._inductor.runtime import autotune_cache as ac
        from torch._inductor.runtime import triton_heuristics as th

        tv = torch.__version__
        applied = []

        # ---- fix 1: trust coordesc entries across candidate-set variations ----
        if tv.startswith(_READ_BEST_VERSIONS):
            load_cached_autotuning = ac._load_cached_autotuning

            def read_best(self, inductor_meta, configs):
                entry = self._read()
                if entry is None:
                    return None
                effective_hash = self.configs_hash
                if entry.get("found_by_coordesc", False) and inductor_meta.get(
                    "coordinate_descent_tuning"
                ):
                    effective_hash = entry.get("configs_hash", effective_hash)
                return load_cached_autotuning(entry, effective_hash, configs, inductor_meta)

            ac.AutotuneCache.read_best = read_best
            applied.append("read_best[2.9-2.13]")

        # ---- fixes 2+3: bundle-restore path adopts and trusts the entry ----
        # 2.12+ ships the upstream fix (pytorch#173391); keep theirs there.
        if tv.startswith(_RECHECK_VERSIONS):
            check_autotune_cache = th.check_autotune_cache
            triton_config_to_hashable = th.triton_config_to_hashable
            dynamo_timed = th.dynamo_timed

            def recheck_autotune_cache(self, reload_kernel_from_src):
                assert self.is_statically_launchable()

                configs = [result.config for result in self.compile_results]
                (cached_configs, _, autotune_cache_info) = check_autotune_cache(
                    configs, self.filename, self.inductor_meta
                )
                self.autotune_cache_info = autotune_cache_info

                # Upstream also requires `len(configs) > 1`, which is never
                # true for bundles that restored exactly one compile result.
                if len(cached_configs) != 1:
                    return
                best_config = cached_configs[0]
                best_config_hash = triton_config_to_hashable(best_config)

                for compile_result in self.compile_results:
                    if triton_config_to_hashable(compile_result.config) == best_config_hash:
                        self.compile_results = [compile_result]
                        # Upstream loses this flag, making run() redo coordesc.
                        if getattr(best_config, "found_by_coordesc", False):
                            compile_result.config.found_by_coordesc = True
                        return

                if getattr(best_config, "found_by_coordesc", False):
                    with dynamo_timed("CachingAutotuner.slow_precompile_config"):
                        if self.fn.fn is None:
                            self.fn = reload_kernel_from_src().fn
                        self.compile_results = [self._precompile_config(best_config)]

            th.CachingAutotuner.recheck_autotune_cache = recheck_autotune_cache
            applied.append("recheck_autotune_cache[2.9-2.11]")

        if applied:
            _INSTALLED = True
            logger.info(
                "[inductor_autotune_fix] installed on torch %s: %s", tv, ", ".join(applied)
            )
        else:
            logger.warning(
                "[inductor_autotune_fix] torch %s not in any patch allowlist, "
                "nothing installed — re-verify defects against this version's "
                "source before extending the allowlists",
                tv,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[inductor_autotune_fix] patch not installed: %r", exc)
