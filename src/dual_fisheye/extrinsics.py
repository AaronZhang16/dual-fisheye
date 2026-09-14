"""Spherical eight-point RANSAC and robust relative-pose refinement; no I/O."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .models import Array


def essential_matrix(first: Array, second: Array) -> Array:
    """Fit E where b2.T E b1=0, using unit bearings without perspective division."""
    design = np.einsum("ni,nj->nij", second, first).reshape(-1, 9)
    _, singular, vt = np.linalg.svd(design, full_matrices=len(design) < 9)
    if singular[7] < singular[0] * 1e-8:
        raise ValueError("Degenerate bearing distribution")
    u, s, vt = np.linalg.svd(vt[-1].reshape(3, 3))
    return u @ np.diag([(s[0] + s[1]) / 2] * 2 + [0.0]) @ vt


def epipolar_error(e: Array, first: Array, second: Array) -> Array:
    """Symmetric sine of angular distance to spherical epipolar planes."""
    n2, n1 = first @ e.T, second @ e
    distance = np.einsum("ni,ni->n", second, n2)
    return distance * np.sqrt(
        0.5
        * (
            1 / np.maximum(np.sum(n1 * n1, axis=1), 1e-20)
            + 1 / np.maximum(np.sum(n2 * n2, axis=1), 1e-20)
        )
    )


def cross_matrix(t: Array) -> Array:
    """Return the matrix representing t cross x."""
    x, y, z = t
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def ray_depths(
    rotation: Array, translation: Array, first: Array, second: Array
) -> tuple[Array, Array]:
    """Triangulate signed distances along rays; negative camera z is allowed."""
    a = first @ rotation.T
    c = np.einsum("ni,ni->n", a, second)
    at, bt = a @ translation, second @ translation
    den = np.maximum(1 - c * c, 1e-12)
    return (-at + c * bt) / den, (-c * at + bt) / den


def decompose(e: Array, first: Array, second: Array) -> tuple[Array, Array]:
    """Select the essential-matrix solution with most positive ray depths."""
    u, _, vt = np.linalg.svd(e)
    u[:, -1] *= np.linalg.det(u)
    vt[-1] *= np.linalg.det(vt)
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    candidates = []
    for rotation in (u @ w @ vt, u @ w.T @ vt):
        for translation in (u[:, 2], -u[:, 2]):
            d1, d2 = ray_depths(rotation, translation, first, second)
            candidates.append((int(np.sum((d1 > 0) & (d2 > 0))), rotation, translation))
    _, rotation, translation = max(candidates, key=lambda item: item[0])
    return rotation, translation


def estimate_extrinsics(
    first: Array,
    second: Array,
    iterations: int = 2000,
    threshold_deg: float = 0.3,
    seed: int = 0,
    min_parallax_deg: float = 0.05,
) -> tuple[Array, Array, dict]:
    """Estimate R21 and unit translation from multi-frame matched bearings.

    Args:
        first: Unit bearings in camera 1, shape (N,3).
        second: Corresponding unit bearings in camera 2.
        iterations: Fixed seeded RANSAC budget.
        threshold_deg: Approximate symmetric epipolar angular threshold.
        seed: Reproducible random seed.
        min_parallax_deg: Reject weak translation observability below this value.

    Returns:
        R21, unit t21 and diagnostics. Translation has no metric scale.
    """
    if len(first) < 16 or first.shape != second.shape:
        raise ValueError("Need at least 16 bearing correspondences")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Non-finite bearings")
    rng = np.random.default_rng(seed)
    threshold = np.sin(np.deg2rad(threshold_deg))
    best = np.zeros(len(first), dtype=bool)
    for _ in range(iterations):
        index = rng.choice(len(first), 8, replace=False)
        try:
            e = essential_matrix(first[index], second[index])
        except ValueError:
            continue
        inside = np.abs(epipolar_error(e, first, second)) < threshold
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < max(16, 0.15 * len(first)):
        raise ValueError(
            "Too few geometric inliers; check texture, overlap and intrinsics"
        )
    e = essential_matrix(first[best], second[best])
    rotation, translation = decompose(e, first[best], second[best])
    _, _, vt = np.linalg.svd(translation.reshape(1, 3))
    tangent = vt[1:].T

    def unpack(x: Array) -> tuple[Array, Array]:
        t = translation + tangent @ x[3:]
        return Rotation.from_rotvec(x[:3]).as_matrix(), t / np.linalg.norm(t)

    def residual(x: Array) -> Array:
        r, t = unpack(x)
        return epipolar_error(cross_matrix(t) @ r, first[best], second[best])

    fit = least_squares(
        residual,
        np.r_[Rotation.from_matrix(rotation).as_rotvec(), 0.0, 0.0],
        loss="soft_l1",
        f_scale=threshold / 2,
        max_nfev=300,
    )
    if not fit.success:
        raise ValueError("Extrinsic refinement failed: " + fit.message)
    rotation, translation = unpack(fit.x)
    errors = np.rad2deg(
        np.arcsin(
            np.clip(
                np.abs(
                    epipolar_error(cross_matrix(translation) @ rotation, first, second)
                ),
                0,
                1,
            )
        )
    )
    d1, d2 = ray_depths(rotation, translation, first, second)
    inliers = (errors < threshold_deg) & (d1 > 0) & (d2 > 0)
    if inliers.sum() < 16:
        raise ValueError("Insufficient positive-depth inliers")
    cos_angle = np.einsum("ni,ni->n", first @ rotation.T, second)
    parallax = float(
        np.median(np.rad2deg(np.arccos(np.clip(cos_angle[inliers], -1, 1))))
    )
    if parallax < min_parallax_deg:
        raise ValueError(
            "Parallax too small to estimate translation reliably; use varied depths"
        )
    return (
        rotation,
        translation,
        {
            "inliers": inliers,
            "errors_deg": errors,
            "inlier_count": int(inliers.sum()),
            "match_count": len(first),
            "median_parallax_deg": parallax,
            "median_error_deg": float(np.median(errors[inliers])),
            "p95_error_deg": float(np.percentile(errors[inliers], 95)),
            "nfev": fit.nfev,
        },
    )
