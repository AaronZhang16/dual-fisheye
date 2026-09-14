"""Thin CLI for synthetic fixtures; install the project with pip install -e . first."""

import argparse
from pathlib import Path
from dual_fisheye.datasets import generate_dataset


def main() -> None:
    """Parse the output project root and call the dataset generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    try:
        generate_dataset(args.root)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Created synthetic fixtures in {args.root.resolve()}")


if __name__ == "__main__":
    main()
