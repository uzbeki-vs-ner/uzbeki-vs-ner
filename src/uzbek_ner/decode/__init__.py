"""Decode postprocess (gold-free span snap and k-fold tuning)."""

from uzbek_ner.decode.kfold import fit_offset_mode, make_folds
from uzbek_ner.decode.snap import snap_entities

__all__ = ["fit_offset_mode", "make_folds", "snap_entities"]
