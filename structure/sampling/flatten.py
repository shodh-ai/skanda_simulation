"""
Utility: flatten a nested dict into a single-level dict with prefixed keys.
Used by GenConfig.to_flat_dict() and SimConfig.to_flat_dict() for CSV export.
"""

from __future__ import annotations
from typing import Any


def flatten_dict(d: dict, prefix: str = "", sep: str = "_") -> dict[str, Any]:
    """
    Recursively flatten a nested dict.

    Examples
    --------
    >>> flatten_dict({"a": {"b": 1}, "c": 2}, prefix="gen")
    {"gen_a_b": 1, "gen_c": 2}

    >>> flatten_dict({"calendering": {"compression_ratio": 0.75}}, prefix="gen")
    {"gen_calendering_compression_ratio": 0.75}

    Parameters
    ----------
    d      : The (possibly nested) dict to flatten.
    prefix : String prepended to every key (empty string = no prefix).
    sep    : Separator between prefix/parent key and child key.

    Returns
    -------
    Flat dict with string keys and scalar values.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key, sep=sep))
        elif isinstance(v, list):
            # Encode lists as semicolon-separated strings for CSV compatibility.
            out[key] = ";".join(str(x) for x in v)
        else:
            out[key] = v
    return out
