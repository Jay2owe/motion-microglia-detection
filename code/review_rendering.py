"""Established full-field issue-review rendering style.

This is the production-neutral home of the visual conventions previously shared by
loading the Issue 008 renderer: non-repeating identity colours, component-centred
labels, fixed whole-movie raw contrast, and a 24-pixel header.
"""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from common import display_raw, label_edges
from pipeline import _motion_rgb


def identity_colour(identity: int) -> np.ndarray:
    hue = (identity * 0.618033988749895) % 1.0
    saturation = 0.62 + 0.08 * (identity % 4)
    value = 0.82 + 0.08 * (identity % 3)
    rgb = colorsys.hsv_to_rgb(hue, min(saturation, 0.9), min(value, 0.98))
    return np.array([round(channel * 255) for channel in rgb], np.uint8)


def annotate_identities(stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Draw every substantial connected component exactly as prior reviews did."""
    result = stack.copy()
    for t in range(len(result)):
        image = Image.fromarray(result[t])
        draw = ImageDraw.Draw(image)
        for identity in sorted(set(map(int, np.unique(labels[t]))) - {0}):
            parts, count = ndi.label(labels[t] == identity,
                                     structure=np.ones((3, 3), np.uint8))
            for component in range(1, count + 1):
                mask = parts == component
                if int(mask.sum()) < 8:
                    continue
                y, x = ndi.center_of_mass(mask)
                colour = tuple(map(int, identity_colour(identity)))
                draw.text((round(x), round(y)), str(identity), fill=colour,
                          stroke_width=2, stroke_fill=(0, 0, 0))
        result[t] = np.asarray(image)
    return result


def unique_outline(raw: np.ndarray, labels: np.ndarray,
                   motion: np.ndarray | None = None) -> np.ndarray:
    out = display_raw(raw) if motion is None else _motion_rgb(raw, motion, labels)
    edge = label_edges(labels, thick=2)
    for identity in sorted(set(map(int, np.unique(labels))) - {0}):
        out[edge & (labels == identity)] = identity_colour(identity)
    return annotate_identities(out, labels)


def header(stack: np.ndarray, title: str, source_offset: int,
           notes: list[str] | None = None, height: int = 24) -> np.ndarray:
    result = np.zeros((len(stack), stack.shape[1] + height,
                       stack.shape[2], 3), np.uint8)
    result[:, height:] = stack
    for frame in range(len(result)):
        image = Image.fromarray(result[frame])
        line = (f"{title} | review {frame + 1} | "
                f"source ImageJ {frame + source_offset + 1}")
        if notes and notes[frame]:
            line += f" | {notes[frame]}"
        ImageDraw.Draw(image).text((4, 4), line, fill=(255, 255, 255))
        result[frame] = np.asarray(image)
    return result
