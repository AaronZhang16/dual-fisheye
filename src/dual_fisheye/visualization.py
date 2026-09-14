"""Diagnostic rendering from already computed results; no optimization."""

import cv2
import numpy as np


def corner_overlay(
    image: np.ndarray, measured: np.ndarray, predicted: np.ndarray | None = None
) -> np.ndarray:
    """Draw measured corners in green, optional projections in red."""
    out = image.copy()
    for x, y in measured:
        cv2.circle(out, (round(x), round(y)), 3, (0, 255, 0), 1)
    if predicted is not None:
        for x, y in predicted:
            cv2.drawMarker(
                out, (round(x), round(y)), (0, 0, 255), cv2.MARKER_CROSS, 7, 1
            )
    return out


def match_overlay(
    first: np.ndarray,
    second: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    inliers: np.ndarray | None = None,
) -> np.ndarray:
    """Draw up to 150 deterministic matches; green inliers, red outliers."""
    h, w = max(first.shape[0], second.shape[0]), first.shape[1]
    out = np.zeros((h, w + second.shape[1], 3), dtype=np.uint8)
    out[: first.shape[0], :w] = first
    out[: second.shape[0], w:] = second
    for j in np.linspace(0, len(p1) - 1, min(150, len(p1)), dtype=int):
        a, b = tuple(np.rint(p1[j]).astype(int)), tuple(
            np.rint(p2[j] + [w, 0]).astype(int)
        )
        color = (0, 255, 0) if inliers is None or inliers[j] else (0, 0, 255)
        cv2.line(out, a, b, color, 1)
    return out
