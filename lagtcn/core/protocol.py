"""Shared Applied Energy experiment-stage protocol helpers."""
from __future__ import annotations


FORMAL_AE_STAGE_PREFIXES = ("ae_final_",)


def is_formal_ae_stage(stage: object) -> bool:
    """Return whether ``stage`` is governed by the frozen AE protocol."""
    value = str(stage or "").strip()
    return value.startswith(FORMAL_AE_STAGE_PREFIXES)
