"""CLI parsing, reproducible run metadata and workflow dispatch."""

import argparse
import logging
import platform
from pathlib import Path
import sys
import cv2
import numpy as np
import scipy
from .io_utils import read_json, write_json
from .models import Camera
from .workflow import calibrate_cameras, calibrate_rig, render_frames


def main() -> int:
    """Execute an offline stage or pipeline; normal pipeline writes only panoramas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["intrinsics", "extrinsics", "stitch", "run"]
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--mode", choices=["normal", "debug"], default="normal")
    parser.add_argument("--output", type=Path, default=Path("outputs/latest"))
    parser.add_argument(
        "--cameras", type=Path, help="Existing intrinsic JSON, relative to project root"
    )
    parser.add_argument(
        "--rig", type=Path, help="Existing extrinsic JSON, relative to project root"
    )
    args = parser.parse_args()
    debug = None
    try:
        config_path = args.config.resolve()
        config = read_json(config_path)
        root = (config_path.parent / config.get("project_root", "..")).resolve()
        output = root / args.output
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                f"Output directory is not empty; choose a new --output: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
        if args.mode == "debug":
            debug = output / "debug"
            debug.mkdir()
            logging.basicConfig(
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(debug / "run.log", encoding="utf-8"),
                ],
                force=True,
            )
            write_json(
                debug / "00_run.json",
                {
                    "config": config,
                    "command": vars(args)
                    | {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "opencv": cv2.__version__,
                },
            )
        else:
            logging.basicConfig(level=logging.ERROR, force=True)
        np.random.seed(config["seed"])
        cv2.setRNGSeed(config["seed"])
        cv2.setNumThreads(1)
        if args.cameras:
            cameras = [
                Camera.from_dict(c) for c in read_json(root / args.cameras)["cameras"]
            ]
        elif args.command in {"run", "intrinsics"}:
            cameras = calibrate_cameras(config, root, debug)
        else:
            raise ValueError("This command requires --cameras")
        if len(cameras) != 2:
            raise ValueError("Exactly two cameras are required")
        if args.command == "intrinsics":
            write_json(
                output / "cameras.json", {"cameras": [c.to_dict() for c in cameras]}
            )
            return 0
        if args.rig:
            rig = read_json(root / args.rig)
        elif args.command in {"run", "extrinsics"}:
            rig = calibrate_rig(config, root, cameras, debug)
        else:
            raise ValueError("stitch requires --rig")
        if args.command == "extrinsics":
            write_json(output / "rig.json", rig)
            return 0
        if debug:
            write_json(
                debug / "cameras.json", {"cameras": [c.to_dict() for c in cameras]}
            )
            write_json(debug / "rig.json", rig)
        render_frames(config, root, cameras, rig, output / "panoramas", debug)
        return 0
    except (ValueError, OSError, KeyError, cv2.error) as error:
        if debug:
            logging.exception("Pipeline failed")
        else:
            print(f"Error: {error}", file=sys.stderr)
        return 2
