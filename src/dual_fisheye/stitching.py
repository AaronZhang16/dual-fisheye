"""Inverse equirectangular mapping for distant scenes; no file operations."""

import cv2
import numpy as np

from .models import Array, Camera


def build_maps(
    first: Camera, second: Camera, rotation: Array, width: int = 2048
) -> tuple[list[Array], list[Array]]:
    """Build reusable float32 maps and angular feather weights.

    Args:
        rotation: R21, mapping camera 1 directions into camera 2.
        width: Equirectangular output width; height is width/2.

    Returns:
        Two maps (H,W,2) and two weights (H,W). Uncovered pixels have zero weight.
        Longitude zero faces camera 1 +z; north is camera 1 -y. Ignores translation.
    """
    if width < 16 or width % 2:
        raise ValueError("Panorama width must be even and at least 16")
    yy, xx = np.mgrid[: width // 2, :width]
    lon = ((xx + 0.5) / width - 0.5) * 2 * np.pi
    lat = (0.5 - (yy + 0.5) / (width // 2)) * np.pi
    rays = np.stack(
        (np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)), -1
    )
    maps, weights = [], []
    for camera, directions in ((first, rays), (second, rays @ rotation.T)):
        uv, valid = camera.project(directions)
        angle = np.arccos(np.clip(directions[..., 2], -1, 1))
        margin = np.deg2rad(camera.fov_deg / 2) - angle
        weight = np.where(valid, np.maximum(margin, 1e-6), 0.0)
        uv[~valid] = -1
        maps.append(uv.astype(np.float32))
        weights.append(weight.astype(np.float32))
    return maps, weights


def stitch(
    first: Array, second: Array, maps: list[Array], weights: list[Array]
) -> tuple[Array, list[Array]]:
    """Remap two BGR images and feather; return panorama and unmixed projections."""
    layers = [
        cv2.remap(im, mp, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        for im, mp in zip((first, second), maps)
    ]
    total = weights[0] + weights[1]
    mixed = sum(
        layer.astype(np.float32) * weight[..., None]
        for layer, weight in zip(layers, weights)
    )
    panorama = mixed / np.maximum(total[..., None], 1e-12)
    return np.clip(np.rint(panorama), 0, 255).astype(np.uint8), layers
