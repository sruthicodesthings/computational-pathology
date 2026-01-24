import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from numerics import integrate_section_areas, radius_limited_nearest, apply_affine


class TestNumerics(unittest.TestCase):
    def test_volume(self):
        self.assertAlmostEqual(integrate_section_areas([0, 2], [2, 4]), 6)

    def test_unsorted(self):
        self.assertAlmostEqual(integrate_section_areas([2, 0], [4, 2]), 6)

    def test_duplicate_position(self):
        with self.assertRaises(ValueError):
            integrate_section_areas([1, 1], [2, 3])

    def test_radius(self):
        self.assertEqual(radius_limited_nearest([0, 0, 0], [[0.1, 0, 0]], .05)[0], None)
        self.assertEqual(radius_limited_nearest([0, 0, 0], [[0.01, 0, 0]], .05)[0], 0)

    def test_affine(self):
        import numpy as np
        t = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]])
        self.assertTrue(np.allclose(apply_affine([[1, 2]], t), [[3, 5]]))

if __name__ == '__main__':
    unittest.main()
