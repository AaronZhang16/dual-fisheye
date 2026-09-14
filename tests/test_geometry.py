"""Synthetic tests with independent geometry and explicit accuracy assertions."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from dual_fisheye.models import Camera
from dual_fisheye.extrinsics import estimate_extrinsics
from dual_fisheye.intrinsics import board_points, estimate_intrinsics
from dual_fisheye.io_utils import paired_paths
from dual_fisheye.stitching import build_maps, stitch


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = Camera(
            np.array([220.0, 220.0, 319.5, 319.5, 0.5, 0.55]), (640, 640), 210.0
        )

    def test_ds_roundtrip_beyond_180(self) -> None:
        theta = np.deg2rad(np.array([0.0, 30.0, 60.0, 89.0, 95.0, 102.0]))
        rays = np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
        pixels, valid = self.camera.project(rays)
        actual, back_valid = self.camera.unproject(pixels)
        self.assertTrue(valid.all() and back_valid.all())
        np.testing.assert_allclose(actual, rays, atol=1e-10)
        _, bad = self.camera.unproject(np.array([[-10.0, -10.0], [10000.0, 10000.0]]))
        self.assertFalse(bad.any())

    def test_pose_outliers_and_negative_z(self) -> None:
        rng = np.random.default_rng(3)
        azimuth = rng.uniform(-np.pi, np.pi, 240)
        radius = rng.uniform(0.8, 3.0, 240)
        points = np.column_stack(
            (
                radius * np.cos(azimuth),
                radius * np.sin(azimuth),
                rng.uniform(-0.12, 0.12, 240),
            )
        )
        r = Rotation.from_euler("xyz", [1.0, 179.0, -2.0], degrees=True).as_matrix()
        t = np.array([0.01, -0.005, 0.08])
        first = points / np.linalg.norm(points, axis=1, keepdims=True)
        second = points @ r.T + t
        second /= np.linalg.norm(second, axis=1, keepdims=True)
        second += rng.normal(0, 2e-5, second.shape)
        second /= np.linalg.norm(second, axis=1, keepdims=True)
        second[:40] = second[40:80]
        actual_r, actual_t, report = estimate_extrinsics(first, second, 800, 0.08, 10)
        self.assertLess(
            np.rad2deg(Rotation.from_matrix(actual_r @ r.T).magnitude()), 0.1
        )
        self.assertGreater(actual_t @ (t / np.linalg.norm(t)), 0.999)
        self.assertGreater(report["inlier_count"], 185)

    def test_intrinsics_held_out_projection(self) -> None:
        objects = board_points(9, 6, 0.025)
        objects -= objects.mean(axis=0)
        observations = []
        rng = np.random.default_rng(4)
        for _ in range(12):
            r = Rotation.from_rotvec(rng.uniform(-0.7, 0.7, 3)).as_matrix()
            p = objects @ r.T + [
                rng.uniform(-0.25, 0.25),
                rng.uniform(-0.25, 0.25),
                rng.uniform(0.4, 0.7),
            ]
            observations.append(self.camera.project(p)[0])
        params = self.camera.params.copy()
        params[:2] *= 1.01
        fitted, report = estimate_intrinsics(
            objects, observations, Camera(params, (640, 640), 210.0), 1000
        )
        self.assertLess(report["rms_px"], 0.03)
        # Evaluate rays not used during fitting, including >90-degree incidence.
        theta = np.deg2rad([10.0, 45.0, 80.0, 98.0])
        rays = np.column_stack((np.sin(theta), np.zeros(4), np.cos(theta)))
        self.assertLess(
            np.max(
                np.linalg.norm(
                    fitted.project(rays)[0] - self.camera.project(rays)[0], axis=1
                )
            ),
            0.5,
        )

    def test_stitch_coverage_and_color(self) -> None:
        r = Rotation.from_euler("y", 180.0, degrees=True).as_matrix()
        maps, weights = build_maps(self.camera, self.camera, r, 256)
        self.assertTrue(np.all(weights[0] + weights[1] > 0))
        im = np.full((640, 640, 3), [20, 100, 200], dtype=np.uint8)
        panorama, _ = stitch(im, im, maps, weights)
        self.assertEqual(panorama.shape, (128, 256, 3))
        np.testing.assert_allclose(
            panorama, np.broadcast_to([20, 100, 200], panorama.shape), atol=0
        )

    def test_reject_missing_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.mkdir()
            b.mkdir()
            (a / "001.png").touch()
            (b / "002.png").touch()
            with self.assertRaises(ValueError):
                paired_paths(a, b)


if __name__ == "__main__":
    unittest.main()
