"""Stitch two or more overlapping photographs into a panorama."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def stitch_images(images: list):
    if len(images) < 2:
        raise ValueError("At least two images are required")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images)
    if status != cv2.Stitcher_OK or panorama is None:
        raise RuntimeError(
            f"OpenCV Stitcher could not create a panorama (status={status}); "
            "try photographs with more overlap and visible features"
        )
    return panorama


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/panorama.jpg"))
    args = parser.parse_args()
    if len(args.images) < 2:
        parser.error("Provide at least two overlapping images")

    images = []
    for path in args.images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            parser.error(f"Cannot read image: {path}")
        images.append(image)
    try:
        result = stitch_images(images)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise OSError(f"Cannot write panorama: {args.output}")
    print(f"Saved panorama: {args.output}")


if __name__ == "__main__":
    main()
