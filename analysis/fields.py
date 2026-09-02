"""Field-level geometry more than one module needs.

Three small routines that were each written for one module and are now wanted
by two or three. They live here rather than being imported across modules by
their private names, and rather than being copied - a copied convolution that
drifts from its original is two modules quietly disagreeing about how big a
neighbourhood is.

Nothing here measures a cell or an object. These are operations on a field:
how far a disc reaches, how much of something lies within that reach, and how
long an unbroken stretch of frames is.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def disc(radius: float) -> np.ndarray:
    """A filled circle of ``radius`` pixels, as a float kernel."""
    span = int(np.ceil(radius))
    grid = np.arange(-span, span + 1)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    return (np.hypot(yy, xx) <= radius).astype(float)


def ground_within(ground: np.ndarray, radius: float) -> np.ndarray:
    """How much of ``ground`` lies within ``radius`` of each pixel of the field.

    One convolution answers it for every pixel at once. Asking per cell-frame
    would be six thousand disc counts to answer a question with a hundred and
    fifty thousand possible answers, and the answer does not depend on which
    cell is asking.

    The result is clipped at zero: an FFT convolution of a binary mask returns
    values a few parts in 1e-12 below zero where the true count is zero, and a
    negative area is not a thing.
    """
    counts = fftconvolve(ground.astype(float), disc(radius), mode="same")
    return np.maximum(counts, 0.0)


def longest_run(frames: np.ndarray) -> int:
    """The longest unbroken stretch of consecutive frame numbers.

    A missing frame breaks the stretch, and that is the point rather than a
    limitation: a cell that was not seen in a frame was not observed to be
    doing anything in it, so bridging the gap would invent the thing being
    counted during a frame where nothing was looked at.
    """
    if not len(frames):
        return 0
    longest = current = 1
    for gap in np.diff(np.sort(frames)):
        current = current + 1 if gap == 1 else 1
        longest = max(longest, current)
    return int(longest)
