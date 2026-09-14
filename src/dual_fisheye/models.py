"""Double Sphere projection; x right, y down, z forward in each camera."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Camera:
    """DS parameters and independently supplied usable angular field of view.

    Args:
        params: [fx, fy, cx, cy, xi, alpha], in pixels except xi and alpha.
        size: Original image (width, height); resizing is not implicit.
        fov_deg: Usable full circular FOV, not sensor diagonal FOV.
    """

    params: Array
    size: tuple[int, int]
    fov_deg: float

    def __post_init__(self) -> None:
        p = np.asarray(self.params, dtype=float)
        if p.shape != (6,) or not np.isfinite(p).all():
            raise ValueError("Camera requires six finite DS parameters")
        if min(p[:2]) <= 0 or not (0 <= p[4] < 1 and 0 < p[5] < 1):
            raise ValueError("Require fx,fy>0, 0<=xi<1, 0<alpha<1")
        if min(self.size) <= 0 or not 0 < self.fov_deg < 360:
            raise ValueError("Invalid image size or FOV")
        object.__setattr__(self, "params", p.copy())

    def project(self, points: Array) -> tuple[Array, NDArray[np.bool_]]:
        """Project camera-frame points, returning pixels and validity mask.

        Args:
            points: Array (..., 3); negative z is supported in valid DS domain.

        Returns:
            Pixels (..., 2) and mask including model domain, FOV and sensor bounds.
            Invalid pixels are finite where possible; always respect the mask.
        """
        x, y, z = np.moveaxis(np.asarray(points, dtype=float), -1, 0)
        fx, fy, cx, cy, xi, alpha = self.params
        d1 = np.sqrt(x * x + y * y + z * z)
        zz = xi * d1 + z
        d2 = np.sqrt(x * x + y * y + zz * zz)
        den = alpha * d2 + (1 - alpha) * zz
        uv = np.stack(
            (
                fx * x / np.maximum(den, 1e-12) + cx,
                fy * y / np.maximum(den, 1e-12) + cy,
            ),
            axis=-1,
        )
        w1 = min(alpha / (1 - alpha), (1 - alpha) / alpha)
        w2 = (w1 + xi) / np.sqrt(2 * w1 * xi + xi * xi + 1)
        valid = (d1 > 1e-12) & (z > -w2 * d1) & (den > 1e-12)
        valid &= z / np.maximum(d1, 1e-12) >= np.cos(np.deg2rad(self.fov_deg / 2))
        valid &= (uv[..., 0] >= 0) & (uv[..., 0] <= self.size[0] - 1)
        valid &= (uv[..., 1] >= 0) & (uv[..., 1] <= self.size[1] - 1)
        return uv, valid & np.isfinite(uv).all(axis=-1)

    def unproject(self, pixels: Array) -> tuple[Array, NDArray[np.bool_]]:
        """Convert original pixels to unit bearings and a domain/FOV mask."""
        p = np.asarray(pixels, dtype=float)
        fx, fy, cx, cy, xi, alpha = self.params
        mx, my = (p[..., 0] - cx) / fx, (p[..., 1] - cy) / fy
        r2 = mx * mx + my * my
        root = 1 - (2 * alpha - 1) * r2
        mz = (1 - alpha * alpha * r2) / (
            alpha * np.sqrt(np.maximum(root, 0)) + 1 - alpha
        )
        root2 = mz * mz + (1 - xi * xi) * r2
        k = (mz * xi + np.sqrt(np.maximum(root2, 0))) / np.maximum(mz * mz + r2, 1e-12)
        b = np.stack((k * mx, k * my, k * mz - xi), axis=-1)
        b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
        uv, valid = self.project(b)
        valid &= (root >= 0) & (root2 >= 0)
        valid &= np.linalg.norm(uv - p, axis=-1) < 1e-5
        return b, valid

    def to_dict(self) -> dict:
        """Serialize parameters with explicit conventions."""
        return {
            "model": "double_sphere",
            "params": self.params.tolist(),
            "size": list(self.size),
            "fov_deg": self.fov_deg,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "Camera":
        """Read a validated DS camera from a JSON-compatible mapping."""
        if value.get("model") != "double_sphere":
            raise ValueError("Only double_sphere is supported")
        return cls(np.asarray(value["params"]), tuple(value["size"]), value["fov_deg"])
