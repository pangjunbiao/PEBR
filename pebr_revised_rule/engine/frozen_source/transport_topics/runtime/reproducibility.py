"""Stable stream identities and process settings; no Python hash() seeding."""
from __future__ import annotations

import hashlib
import json
import os


def stable_seed(base_seed: int, namespace: str, *identifiers) -> int:
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    data = json.dumps([base_seed, namespace, list(identifiers)], ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def configure_process(runtime: dict, model_seeds: list[int] | None = None) -> dict:
    values = {"OMP_NUM_THREADS": str(runtime["cpu_threads"]),
              "MKL_NUM_THREADS": str(runtime["cpu_threads"]),
              "OPENBLAS_NUM_THREADS": str(runtime["cpu_threads"]),
              "NUMEXPR_NUM_THREADS": str(runtime["cpu_threads"]),
              "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1",
              "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    for name, value in values.items():
        os.environ[name] = value
    return {"settings": values,
            "python_hash_seed_at_launch": os.environ.get("PYTHONHASHSEED"),
            "stable_seed_policy": "SHA256 over explicit namespace and identifiers; never hash()",
            "model_seeds": list(range(5)) if model_seeds is None else list(model_seeds)}
