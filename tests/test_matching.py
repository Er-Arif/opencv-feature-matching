import unittest

import numpy as np

from feature_matching import match_keypoints
from main import stitch_images


class MatchingTests(unittest.TestCase):
    def test_matches_identical_textured_images(self):
        rng = np.random.default_rng(19)
        image = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        visualization, count = match_keypoints(image, image)
        self.assertGreater(count, 0)
        self.assertEqual(visualization.shape[0], 256)

    def test_blank_images_fail_helpfully(self):
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            match_keypoints(blank, blank)

    def test_stitching_requires_two_inputs(self):
        with self.assertRaises(ValueError):
            stitch_images([])


if __name__ == "__main__":
    unittest.main()
