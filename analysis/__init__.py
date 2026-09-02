"""Downstream analysis package for the Motion microglia pipeline.

Takes accepted identity outlines plus the registered raw signal they were built
from and produces every measurement, table and figure a user needs, without
touching or recomputing the tracking result.

The tracking pipeline in ``code/`` answers *which pixels are which cell*.
This package answers *what those cells did*.
"""

__version__ = "0.1.0"

from analysis.registry import MeasurementContext, get_module, list_modules, register  # noqa: F401
