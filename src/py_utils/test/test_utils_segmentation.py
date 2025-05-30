import unittest

import numpy as np

import src.utils_segmentation as utils_segmentation


class TestSegmentedOperations(unittest.TestCase):

    def test_segmented_sum(self):
        x = np.array([1, 2, 3, 4, 5])
        s_ind = np.array([0, 3])
        e_ind = np.array([3, 5])
        expected = np.array([6, 9])  # 1+2+3 and 4+5

        np.testing.assert_array_equal(
            utils_segmentation.segmented_sum(x, s_ind, e_ind), expected
        )

    def test_segmented_count(self):
        x = np.array([1, 2, 3, 4, 5])
        s_ind = np.array([0, 3])
        e_ind = np.array([3, 5])

        # Three elements in first segment, two in second
        expected = np.array([3, 2])

        np.testing.assert_array_equal(
            utils_segmentation.segmented_count(x, s_ind, e_ind), expected
        )

    def test_segmented_mean(self):
        x = np.array([1, 2, 3, 4, 5])
        s_ind = np.array([0, 3])
        e_ind = np.array([3, 5])

        # Mean of 1,2,3 and 4,5
        expected = np.array([2, 4.5])

        np.testing.assert_array_equal(
            utils_segmentation.segmented_mean(x, s_ind, e_ind), expected
        )

    def test_error_conditions(self):
        x = np.array([1, 2, 3])
        s_ind = np.array([0, 2])
        e_ind = np.array([2, 4])  # Invalid e_ind

        with self.assertRaises(ValueError):
            utils_segmentation.segmented_sum(x, s_ind, e_ind)

        with self.assertRaises(ValueError):
            utils_segmentation.segmented_count(x, s_ind, e_ind)

        with self.assertRaises(ValueError):
            utils_segmentation.segmented_mean(x, s_ind, e_ind)


if __name__ == "__main__":
    unittest.main()
