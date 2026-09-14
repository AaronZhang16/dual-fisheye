"""SIFT matching on original fisheye images; geometry stays in unit bearings."""

import cv2
import numpy as np

from .models import Array, Camera


def overlap_mask(
    camera: Camera, other: Camera, rotation: Array, margin_deg: float = 10.0
) -> Array:
    """Find a generous directional overlap using an approximate rotation.

    Args:
        rotation: Maps this camera's column vectors to the other camera.
        margin_deg: Angular search allowance for installation errors.

    Returns:
        uint8 image mask. Occlusions and finite-depth overlap are not predicted.
    """
    w, h = camera.size
    yy, xx = np.mgrid[:h, :w]
    rays, valid = camera.unproject(np.stack((xx, yy), axis=-1))
    remote = rays @ rotation.T
    valid &= remote[..., 2] > np.cos(
        np.deg2rad(min(other.fov_deg / 2 + margin_deg, 179))
    )
    return valid.astype(np.uint8) * 255


def match_features(
    first: Array,
    second: Array,
    masks: tuple[Array, Array],
    max_features: int = 5000,
    ratio: float = 0.75,
) -> tuple[Array, Array]:
    """Return mutual ratio-tested SIFT correspondences in original pixel units."""
    detector = cv2.SIFT_create(nfeatures=max_features)
    k1, d1 = detector.detectAndCompute(
        cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), masks[0]
    )
    k2, d2 = detector.detectAndCompute(
        cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), masks[1]
    )
    if d1 is None or d2 is None or min(len(d1), len(d2)) < 2:
        return np.empty((0, 2)), np.empty((0, 2))

    def accepted(a: Array, b: Array) -> dict[int, int]:
        pairs = cv2.BFMatcher().knnMatch(a, b, k=2)
        return {
            m.queryIdx: m.trainIdx
            for pair in pairs
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < ratio * n.distance
        }

    forward, backward = accepted(d1, d2), accepted(d2, d1)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    return (
        np.asarray([k1[i].pt for i, _ in pairs], dtype=float).reshape(-1, 2),
        np.asarray([k2[j].pt for _, j in pairs], dtype=float).reshape(-1, 2),
    )
