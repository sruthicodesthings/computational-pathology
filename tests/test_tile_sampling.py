import unittest
import numpy as np
from src.tile_sampling import sample_tiles


class TileSamplingTests(unittest.TestCase):
    def setUp(self):
        self.masks = {
            'manual': np.array([[1, 0], [0, 2]]),
            'classifier': np.array([[3, 3], [0, 0]]),
            'hard1': np.array([[2, 2], [0, 0]]),
            'hard2': np.array([[2, 0], [0, 0]]),
            'dark': np.zeros((2, 2), dtype=int),
        }

    def test_all_positive_retained_background_excluded(self):
        r = sample_tiles(self.masks, target_negative_to_positive=10)
        self.assertIn('manual', r.selected_ids)
        self.assertIn('classifier', r.selected_ids)
        self.assertNotIn('dark', r.selected_ids)
        self.assertEqual(len(r.negative_ids), 2)

    def test_ratio_counts_all_pixels(self):
        r = sample_tiles(self.masks, target_negative_to_positive=1)
        self.assertEqual(r.counts, (4, 1, 1, 2))
        self.assertAlmostEqual(r.negative_to_positive_ratio, 5 / 3)
        self.assertEqual(r.negative_ids, ())

    def test_reproducible(self):
        a = sample_tiles(self.masks, target_negative_to_positive=3, seed=7)
        b = sample_tiles(self.masks, target_negative_to_positive=3, seed=7)
        self.assertEqual(a, b)

    def test_unreachable_target(self):
        r = sample_tiles(self.masks, target_negative_to_positive=100)
        self.assertFalse(r.target_reached)

    def test_reject_bad_labels(self):
        with self.assertRaises(ValueError):
            sample_tiles({'bad': np.array([[4]])}, target_negative_to_positive=1)

    def test_requires_positives(self):
        with self.assertRaises(ValueError):
            sample_tiles({'hard': np.array([[2]])}, target_negative_to_positive=1)


if __name__ == '__main__':
    unittest.main()
