"""Offline orchestration and per-stage diagnostic export."""

import logging
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .extrinsics import estimate_extrinsics
from .features import match_features, overlap_mask
from .intrinsics import board_points, detect_board, estimate_intrinsics, initial_camera
from .io_utils import image_paths, paired_paths, read_image, write_image, write_json
from .models import Camera
from .stitching import build_maps, stitch
from .visualization import corner_overlay, match_overlay

LOG = logging.getLogger(__name__)


def calibrate_cameras(config: dict, root: Path, debug: Path | None) -> list[Camera]:
    """Detect boards and fit cameras independently, optionally exporting each stage."""
    s = config["intrinsics"]
    shape = tuple(s["board_inner_corners"])
    objects = board_points(*shape, s["square_size_m"])
    cameras = []
    for i, folder in enumerate(s["folders"]):
        paths = image_paths(root / folder)[: s["max_frames"]]
        observations, images, accepted, rejected = [], [], [], []
        size = None
        for path in paths:
            image = read_image(path, size)
            size = (image.shape[1], image.shape[0])
            points = detect_board(image, shape)
            if points is None:
                rejected.append(path.name)
                continue
            observations.append(points)
            images.append(image)
            accepted.append(path.name)
            if debug:
                write_image(
                    debug / f"01_corners/cam{i+1}/{path.stem}.png",
                    corner_overlay(image, points),
                )
        if debug:
            write_json(
                debug / f"01_corners/cam{i+1}/detections.json",
                {"accepted": accepted, "rejected": rejected, "corners": observations},
            )
        params = s["seed_params"][i]
        seed = (
            Camera(np.array(params), size, s["fov_deg"][i])
            if params
            else initial_camera(size, s["fov_deg"][i])
        )
        LOG.info("Camera %s: fitting %s boards", i + 1, len(observations))
        camera, report = estimate_intrinsics(
            objects, observations, seed, s["max_nfev"], s["loss_px"]
        )
        if report["rms_px"] > s["max_rms_px"]:
            raise ValueError(
                f"Camera {i+1} RMS exceeds configured limit: {report['rms_px']:.3f}px"
            )
        cameras.append(camera)
        if debug:
            write_json(
                debug / f"02_intrinsics/cam{i+1}.json",
                {"camera": camera.to_dict(), **report},
            )
            for name, image, measured, predicted in zip(
                accepted, images, observations, report["predictions"]
            ):
                write_image(
                    debug / f"02_intrinsics/cam{i+1}/{Path(name).stem}.png",
                    corner_overlay(image, measured, predicted),
                )
    return cameras


def calibrate_rig(
    config: dict, root: Path, cameras: list[Camera], debug: Path | None
) -> dict:
    """Pool paired-frame bearings and estimate one shared rigid transform."""
    s = config["extrinsics"]
    rotation = Rotation.from_euler(
        "xyz", s["initial_rotation_xyz_deg"], degrees=True
    ).as_matrix()
    masks = (
        overlap_mask(cameras[0], cameras[1], rotation, s["overlap_margin_deg"]),
        overlap_mask(cameras[1], cameras[0], rotation.T, s["overlap_margin_deg"]),
    )
    if debug:
        for i, mask in enumerate(masks):
            write_image(debug / f"03_overlap/cam{i+1}.png", mask)
    frames = paired_paths(*(root / f for f in s["folders"]))[: s["max_frames"]]
    rays1, rays2, records = [], [], []
    for a, b in frames:
        images = read_image(a, cameras[0].size), read_image(b, cameras[1].size)
        p1, p2 = match_features(*images, masks, s["max_features"], s["ratio"])
        b1, v1 = cameras[0].unproject(p1)
        b2, v2 = cameras[1].unproject(p2)
        valid = v1 & v2
        p1, p2, b1, b2 = p1[valid], p2[valid], b1[valid], b2[valid]
        bins = ((np.arctan2(b1[:, 1], b1[:, 0]) + np.pi) / (2 * np.pi) * 12).astype(
            int
        ) % 12
        keep = np.concatenate(
            [np.flatnonzero(bins == j)[: s["per_bin_cap"]] for j in range(12)]
        )
        p1, p2, b1, b2 = p1[keep], p2[keep], b1[keep], b2[keep]
        rays1.append(b1)
        rays2.append(b2)
        records.append((a, b, p1, p2))
        LOG.info("Pair %s: %s filtered matches", a.stem, len(b1))
        if debug:
            write_image(
                debug / f"04_matches/{a.stem}.png", match_overlay(*images, p1, p2)
            )
            target = debug / f"05_bearings/{a.stem}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target, pixels1=p1, pixels2=p2, bearings1=b1, bearings2=b2
            )
    r, unit_t, report = estimate_extrinsics(
        np.concatenate(rays1),
        np.concatenate(rays2),
        s["ransac_iterations"],
        s["threshold_deg"],
        config["seed"],
        s["min_parallax_deg"],
    )
    baseline = s["baseline_m"]
    if baseline is not None and baseline <= 0:
        raise ValueError("baseline_m must be positive or null")
    rig = {
        "convention": "X2 = R21 @ X1 + t21",
        "R21": r.tolist(),
        "t21_direction": unit_t.tolist(),
        "baseline_m": baseline,
        "t21_m": (unit_t * baseline).tolist() if baseline is not None else None,
    }
    if debug:
        write_json(debug / "06_extrinsics/result.json", {"rig": rig, **report})
        offset = 0
        for a, b, p1, p2 in records:
            inside = report["inliers"][offset : offset + len(p1)]
            write_image(
                debug / f"06_extrinsics/inliers/{a.stem}.png",
                match_overlay(read_image(a), read_image(b), p1, p2, inside),
            )
            offset += len(p1)
    return rig


def render_frames(
    config: dict,
    root: Path,
    cameras: list[Camera],
    rig: dict,
    output: Path,
    debug: Path | None,
) -> None:
    """Render all paired frames using one cached infinite-depth mapping."""
    s = config["stitching"]
    rotation = np.asarray(rig["R21"], dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or np.linalg.det(rotation) < 0.999
    ):
        raise ValueError("R21 is not a proper rotation matrix")
    maps, weights = build_maps(*cameras, rotation, s["width"])
    if debug:
        target = debug / "07_mapping/maps.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target, map1=maps[0], map2=maps[1], weight1=weights[0], weight2=weights[1]
        )
        covered = weights[0] + weights[1] > 0
        write_image(debug / "07_mapping/coverage.png", covered.astype(np.uint8) * 255)
        write_json(
            debug / "07_mapping/report.json",
            {
                "coverage_fraction": float(covered.mean()),
                "projection": "equirectangular_infinite_depth",
            },
        )
    for a, b in paired_paths(*(root / f for f in s["folders"])):
        panorama, layers = stitch(
            read_image(a, cameras[0].size),
            read_image(b, cameras[1].size),
            maps,
            weights,
        )
        write_image(output / f"{a.stem}.png", panorama)
        if debug:
            for i, layer in enumerate(layers):
                write_image(debug / f"08_projection/{a.stem}_cam{i+1}.png", layer)
        LOG.info("Rendered %s", a.stem)
