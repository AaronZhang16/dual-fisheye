"""Generate a complete synthetic dataset using reusable renderers."""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io_utils import read_json, write_json, write_image
from .models import Camera
from .synthetic import render_board, render_room, room_textures


def generate_dataset(root: Path) -> None:
    """Generate distinct synthetic calibration and stitch folders; never real data."""
    root = root.resolve()
    target = root / "data/synthetic"
    if target.exists():
        raise ValueError(
            "data/synthetic exists; preserve it or rename it before regenerating"
        )
    camera = Camera(
        np.array([235.0, 237.0, 319.5, 319.5, 0.5, 0.55]), (640, 640), 210.0
    )
    config = read_json(root / "configs/default.json")
    config["intrinsics"]["folders"] = [
        f"data/synthetic/intrinsics/cam{i}" for i in (1, 2)
    ]
    config["intrinsics"]["fov_deg"] = [210.0, 210.0]
    config["intrinsics"]["square_size_m"] = 0.035
    config["intrinsics"]["seed_params"] = [
        camera.params.tolist(),
        camera.params.tolist(),
    ]
    config["extrinsics"]["folders"] = [
        f"data/synthetic/extrinsics/cam{i}" for i in (1, 2)
    ]
    config["extrinsics"]["baseline_m"] = 0.12
    config["stitching"]["folders"] = [f"data/synthetic/stitch/cam{i}" for i in (1, 2)]
    config["stitching"]["width"] = 1024
    for i in range(16):
        angle = np.deg2rad((i % 4) * 23.0)
        azimuth = i * 2.399963
        center = np.array(
            [
                np.sin(angle) * np.cos(azimuth),
                np.sin(angle) * np.sin(azimuth),
                np.cos(angle),
            ]
        )
        rotation = Rotation.align_vectors(center[None], np.array([[0.0, 0.0, 1.0]]))[
            0
        ].as_matrix()
        rotation = (
            rotation
            @ Rotation.from_euler(
                "xyz", [(-1) ** i * 12, 10, i * 7], degrees=True
            ).as_matrix()
        )
        translation = center * (0.40 + 0.03 * (i % 3)) - rotation @ np.array(
            [0.14, 0.0875, 0.0]
        )
        image = render_board(camera, rotation, translation)
        for folder in config["intrinsics"]["folders"]:
            write_image(root / folder / f"{i:03d}.png", image)
    r = Rotation.from_euler("xyz", [0.5, 179.0, -0.5], degrees=True).as_matrix()
    t = np.array([0.0, 0.0, 0.12])
    for j in range(3):
        textures = room_textures(100 + j)
        images = [
            render_room(camera, np.eye(3), np.zeros(3), textures),
            render_room(camera, r.T, -r.T @ t, textures),
        ]
        for stage in ("extrinsics", "stitching"):
            for folder, image in zip(config[stage]["folders"], images):
                write_image(root / folder / f"{j:03d}.png", image)
    write_json(root / "configs/synthetic.json", config)
    write_json(
        target / "truth.json", {"cameras": [camera.to_dict()] * 2, "R21": r, "t21_m": t}
    )
