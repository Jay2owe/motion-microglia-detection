"""What every panel gives back.

Its own module rather than ``panels/__init__`` because every panel file imports
it and ``__init__`` imports every panel file, and rather than ``_schema``
because a panel must not know what a figure is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

__all__ = ["PanelResult"]


@dataclass(frozen=True)
class PanelResult:
    """The table a panel drew, and anything it made along the way.

    One type rather than six because the caller's first question is always the
    same - what exactly went on the page - and until this existed every builder
    answered it by rebuilding the table from the inputs it had just passed in.
    Two tables that should be identical, computed twice, in thirty-six places.

    ``data`` is one row per mark, holding what a reader would need to redraw it:
    a trace returns its points, a histogram its bins and counts, a raster the
    long form of its matrix. For a strip of image tiles it is one row per tile -
    which frame, at what contrast - and not a copy of the pixels, which are
    already copied and hashed under the bundle's ``data/src``.

    ``axes`` is for a panel that makes axes the caller did not give it, and
    ``extra`` for a value the caller needs and cannot recompute, such as the
    contrast limits a strip chose so that a second strip can match them.
    """

    data: pd.DataFrame
    axes: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
