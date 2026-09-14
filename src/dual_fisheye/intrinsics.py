"""Checkerboard initialization and joint DS/board-pose optimization; no I/O."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix

from .models import Array, Camera


def board_points(columns: int, rows: int, square_m: float) -> Array:
    """Create row-major inner-corner coordinates in metres."""
    if min(columns, rows) < 3 or square_m <= 0:
        raise ValueError("Use at least 3x3 inner corners and positive square size")
    xy = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_m
    return np.column_stack((xy, np.zeros(len(xy))))


def detect_board(image: Array, shape: tuple[int, int]) -> Array | None:
    """Detect subpixel checkerboard corners in the original image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(
        gray, shape, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    )
    return corners.reshape(-1, 2).astype(float) if ok else None


def initial_camera(size: tuple[int, int], fov_deg: float) -> Camera:
    """Initialize DS assuming the circular image diameter fills the short side.

    This is an initialization only; load a measured seed for cropped circles.
    """
    theta = np.deg2rad(fov_deg / 2)
    xi, alpha = 0.5, 0.5
    z = xi + np.cos(theta)
    radius = np.sin(theta) / (
        alpha * np.sqrt(np.sin(theta) ** 2 + z * z) + (1 - alpha) * z
    )
    f = (min(size) - 1) / (2 * radius)
    return Camera(
        np.array([f, f, (size[0] - 1) / 2, (size[1] - 1) / 2, xi, alpha]), size, fov_deg
    )


def initial_board_pose(camera: Camera, objects: Array, pixels: Array) -> Array:
    """Initialize board pose through a locally oriented virtual pinhole camera."""
    rays, valid = camera.unproject(pixels)
    if not valid.all():
        raise ValueError("Board outside seed camera domain; supply a better seed/FOV")
    center = rays.mean(axis=0)
    center /= np.linalg.norm(center)
    q = Rotation.align_vectors(center[None], np.array([[0.0, 0.0, 1.0]]))[0].as_matrix()
    local = rays @ q
    if np.min(local[:, 2]) <= 0.05:
        raise ValueError("Board spans too wide an angle for pose initialization")
    normalized = np.ascontiguousarray(local[:, :2] / local[:, 2:])
    ok, r, t = cv2.solvePnP(
        objects, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise ValueError("Board pose initialization failed")
    return np.r_[
        Rotation.from_matrix(q @ cv2.Rodrigues(r)[0]).as_rotvec(), q @ t.ravel()
    ]


def estimate_intrinsics(
    objects: Array,
    observations: list[Array],
    seed: Camera,
    max_nfev: int = 300,
    loss_px: float = 1.0,
) -> tuple[Camera, dict]:
    """Fit six DS parameters and one rigid board pose per image.

    Args:
        objects: Known checkerboard coordinates (N,3), in metres.
        observations: Original-image corners, all in the same board ordering.
        seed: Approximate camera covering every detected board.
        max_nfev: Optimizer evaluation budget.
        loss_px: Soft-L1 robust loss scale in pixels.

    Returns:
        Camera and diagnostics containing poses, predictions and per-view errors.
    """
    if len(observations) < 6:
        raise ValueError("At least six valid, varied checkerboard views are required")
    poses = [initial_board_pose(seed, objects, p) for p in observations]
    x0 = np.r_[seed.params, np.concatenate(poses)]
    w, h = seed.size
    lower = np.r_[[1.0, 1.0, 0.0, 0.0, 0.0, 0.01], np.full(len(x0) - 6, -np.inf)]
    upper = np.r_[
        [10 * w, 10 * h, w - 1, h - 1, 0.99, 0.99], np.full(len(x0) - 6, np.inf)
    ]

    def predictions(x: Array) -> tuple[list[Array], list[Array]]:
        camera = Camera(x[:6], seed.size, seed.fov_deg)
        projected, validity = [], []
        for pose in x[6:].reshape(-1, 6):
            points = objects @ Rotation.from_rotvec(pose[:3]).as_matrix().T + pose[3:]
            uv, valid = camera.project(points)
            projected.append(uv)
            validity.append(valid)
        return projected, validity

    def residual(x: Array) -> Array:
        projected, _ = predictions(x)
        return np.concatenate(
            [(a - b).ravel() for a, b in zip(projected, observations)]
        )

    count = len(objects) * 2
    sparsity = lil_matrix((count * len(observations), len(x0)), dtype=int)
    for i in range(len(observations)):
        sparsity[i * count : (i + 1) * count, :6] = 1
        sparsity[i * count : (i + 1) * count, 6 + 6 * i : 12 + 6 * i] = 1
    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        jac_sparsity=sparsity,
        tr_options=dict(atol=1e-12, btol=1e-12, maxiter=500),
        loss="soft_l1",
        f_scale=loss_px,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    projected, valid = predictions(result.x)
    if not result.success or not all(v.all() for v in valid):
        raise ValueError("Intrinsic fit failed/domain invalid: " + result.message)
    errors = [np.linalg.norm(a - b, axis=1) for a, b in zip(projected, observations)]
    return Camera(result.x[:6], seed.size, seed.fov_deg), {
        "rms_px": float(np.sqrt(np.mean(np.concatenate(errors) ** 2))),
        "per_view_rms_px": [float(np.sqrt(np.mean(e * e))) for e in errors],
        "p95_px": float(np.percentile(np.concatenate(errors), 95)),
        "poses": result.x[6:].reshape(-1, 6),
        "predictions": projected,
        "nfev": result.nfev,
        "message": result.message,
    }
