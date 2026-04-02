from __future__ import annotations

from typing import Any

import numpy as np
import quapy as qp


def to_serializable(obj: Any) -> Any:
    """
    Recursively convert objects containing NumPy types into JSON-serializable
    Python-native types.
    """
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_quantifier_class(qid: str):
    """
    Resolve a QuaPy quantifier class by identifier.
    """
    if qid.startswith("KDEy"):
        return getattr(qp.method._kdey, qid)
    try:
        q_class = getattr(qp.method.aggregative, qid)
    except AttributeError:
        raise ValueError(f"Quantifier {qid} not found") from None
    return q_class
