"""Compatibility names for analysis now shipped in PyMicroglia.

Use ``pymicroglia`` for new code. The Motion repository owns tracking.
"""
import warnings
from pymicroglia import measure, states, clustering, workbench, tracking

circadian = workbench
warnings.warn('Motion analysis moved to PyMicroglia; use pymicroglia for new code.',
              DeprecationWarning, stacklevel=2)
