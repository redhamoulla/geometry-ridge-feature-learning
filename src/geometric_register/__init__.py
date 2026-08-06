"""Conditional geometric register utilities."""
from .core import (
    whiten_triplet, orbit_signature, contraction_operator,
    information_gain, local_displacement, register_diagnostics,
)
from .robust import fit_bgr, ridge_fit

__all__ = [
    "whiten_triplet", "orbit_signature", "contraction_operator",
    "information_gain", "local_displacement", "register_diagnostics",
    "fit_bgr", "ridge_fit",
]
