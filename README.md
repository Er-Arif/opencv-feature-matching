# OpenCV Feature Matching & Panorama Stitching

A hands-on computer-vision project demonstrating **ORB features**, Hamming-distance descriptor matching, and multi-image panorama stitching.

## Features

- ORB keypoint detection and binary descriptors
- Brute-force Hamming matcher with cross-check
- Optional visualization of the strongest matching keypoints
- OpenCV's modern `Stitcher_create` API for panorama creation
- Command-line paths instead of hardcoded local Windows screenshot locations

The original exploratory script remains in `exp1.py`. A previously generated `panorama.jpg` is included as an archival output; the original source screenshots are not included.

## Run locally

Python 3.10+:

```bash
python -m pip install -r requirements.txt

# Stitch two or more images with overlapping scenes
python main.py left.jpg right.jpg --output output/panorama.jpg

# Visualize ORB feature matches for a pair of images
python feature_matching.py left.jpg right.jpg --output output/matches.jpg --max-matches 30

# Run the feature-matching smoke tests
python -m unittest discover -s tests -v
```

## Results and limits

Stitching works best with photographs of a scene that overlap, contain textured features, and have sufficiently consistent exposure. Some pairs cannot be stitched; the CLI reports that failure rather than fabricating a result. The matcher output is a visualization, not a geometric accuracy score.

This is a learning experiment using traditional computer vision, **not** neural-network-based panorama generation.
