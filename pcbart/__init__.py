"""PCB art board toolkit: generate EasyEDA Pro (``.epro2``) projects.

The package works in millimetres everywhere. The only place the on-disk unit
appears is :mod:`pcbart.units`.
"""

from __future__ import annotations

__version__ = "0.1.2"

from .epru import EproArchive, EproDocument, Record, split_record_line
from .units import EPRU_UNIT_MM, mm_to_unit, unit_to_mm

__all__ = [
    "__version__",
    "EproArchive",
    "EproDocument",
    "Record",
    "split_record_line",
    "EPRU_UNIT_MM",
    "mm_to_unit",
    "unit_to_mm",
]
