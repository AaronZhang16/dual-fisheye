"""Synthetic image fixtures, separate from calibration and production algorithms."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .models import Camera


def pixel_rays(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Return dense original-image unit rays and valid mask."""
    yy, xx = np.mgrid[: camera.size[1], : camera.size[0]]
    return camera.unproject(np.stack((xx, yy), -1))


def render_board(
    camera: Camera,
    rotation: np.ndarray,
    translation: np.ndarray,
    columns: int = 9,
    rows: int = 6,
    square: float = 0.035,
) -> np.ndarray:
    """Render a planar checkerboard by ray-plane intersection, not point drawing."""
    rays, valid = pixel_rays(camera)
    normal = rotation[:, 2]
    dot = rays @ normal
    depth = (translation @ normal) / np.where(np.abs(dot) > 1e-8, dot, 1e-8)
    local = (rays * depth[..., None] - translation) @ rotation
    x, y = local[..., 0], local[..., 1]
    inside = (
        valid
        & (depth > 0)
        & (x >= -square)
        & (x < columns * square)
        & (y >= -square)
        & (y < rows * square)
    )
    color = np.where(
        (np.floor(x / square) + np.floor(y / square)).astype(int) % 2 == 0, 235, 20
    )
    gray = np.where(inside, color, np.where(valid, 125, 0)).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def room_textures(seed: int) -> list[np.ndarray]:
    """Generate six uniquely textured surfaces with reproducible SIFT-scale detail."""
    rng = np.random.default_rng(seed)
    textures = []
    for _ in range(6):
        im = cv2.resize(rng.integers(0, 256, (100, 100, 3), dtype=np.uint8), (800, 800))
        for _ in range(250):
            center = tuple(rng.integers(10, 790, 2).tolist())
            cv2.circle(
                im,
                center,
                int(rng.integers(3, 14)),
                tuple(rng.integers(0, 256, 3).tolist()),
                -1,
            )
        textures.append(im)
    return textures


def render_room(
    camera: Camera,
    world_rotation: np.ndarray,
    origin: np.ndarray,
    textures: list[np.ndarray],
) -> np.ndarray:
    """Ray-trace a textured box with finite depth from a given optical center."""
    rays, valid = pixel_rays(camera)
    rays = rays @ world_rotation.T
    limits = np.array([1.5, 2.0, 2.5])
    hit = np.full(valid.shape, np.inf)
    output = np.zeros((*valid.shape, 3), dtype=np.uint8)
    for axis in range(3):
        other = [j for j in range(3) if j != axis]
        for side, sign in enumerate((-1, 1)):
            component = rays[..., axis]
            distance = (sign * limits[axis] - origin[axis]) / np.where(
                np.abs(component) > 1e-9, component, 1e-9
            )
            p = origin + distance[..., None] * rays
            mask = valid & (distance > 0) & (distance < hit)
            mask &= np.all(np.abs(p[..., other]) <= limits[other] + 1e-8, axis=-1)
            uv = ((p[..., other] / limits[other] + 1) * 0.5 * 799).astype(np.float32)
            uv = np.clip(uv, -1000, 1800)
            layer = cv2.remap(textures[2 * axis + side], uv, None, cv2.INTER_LINEAR)
            output[mask] = layer[mask]
            hit[mask] = distance[mask]
    return output
