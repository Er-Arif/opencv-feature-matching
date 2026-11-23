"""Visualize ORB keypoints matched between two images."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def match_keypoints(left, right, max_matches: int = 30):
    if left is None or right is None:
        raise ValueError("Two readable images are required")
    if max_matches < 1:
        raise ValueError("max_matches must be positive")
    detector = cv2.ORB_create(nfeatures=1500)
    keypoints_a, descriptors_a = detector.detectAndCompute(left, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(right, None)
    if descriptors_a is None or descriptors_b is None:
        raise ValueError("Images do not contain enough ORB-detectable features")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(descriptors_a, descriptors_b), key=lambda match: match.distance)
    if not matches:
        raise ValueError("No ORB matches found")
    strongest = matches[:max_matches]
    preview = cv2.drawMatches(
        left, keypoints_a, right, keypoints_b, strongest, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return preview, len(strongest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/matches.jpg"))
    parser.add_argument("--max-matches", type=int, default=30)
    args = parser.parse_args()

    left = cv2.imread(str(args.left))
    right = cv2.imread(str(args.right))
    try:
        result, count = match_keypoints(left, right, args.max_matches)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise OSError(f"Cannot write feature-match visualization: {args.output}")
    print(f"Saved {count} ORB matches: {args.output}")


if __name__ == "__main__":
    main()
