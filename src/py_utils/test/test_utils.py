import unittest

import numpy as np

import src.utils as utils


class TestCombineMeanVar(unittest.TestCase):

    def test_combine_two_non_zero(self):
        A = np.random.rand(10)
        B = np.random.rand(20)

        mean = np.mean(np.hstack([A, B]))
        var = np.var(np.hstack([A, B]))

        result = utils.combine_two_mean_var(
            (10, np.mean(A), np.var(A)), (20, np.mean(B), np.var(B))
        )
        expect = (30, mean, var)

        for r, e in zip(result, expect):
            self.assertAlmostEqual(r, e)

    def test_combine_with_zero(self):
        result = utils.combine_two_mean_var((0, 0, 0), (15, 3.5, 1.75))
        expect = (15, 3.5, 1.75)

        self.assertTupleEqual(result, expect)

    def test_combine_zero_with_non_zero(self):
        result = utils.combine_two_mean_var((15, 3.5, 1.75), (0, 0, 0))
        expect = (15, 3.5, 1.75)
        self.assertTupleEqual(result, expect)

    def test_combine_multiple(self):
        A = np.random.rand(10)
        B = np.random.rand(20)
        C = np.random.rand(30)

        mean = np.mean(np.hstack([A, B, C]))
        var = np.var(np.hstack([A, B, C]))

        result = utils.combine_mean_var(
            (10, np.mean(A), np.var(A)),
            (20, np.mean(B), np.var(B)),
            (30, np.mean(C), np.var(C)),
        )
        expect = (60, mean, var)

        for r, e in zip(result, expect):
            self.assertAlmostEqual(r, e)

    def test_combine_none(self):
        self.assertListEqual(utils.combine_mean_var(), [0, 0, 0])


class TestPointsInBoundingBox(unittest.TestCase):

    def test_points_inside_bounding_box(self):

        bb_vertices = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, -1],
                [1, -1, 1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, -1],
                [-1, -1, 1],
            ]
        )

        points = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        expected = np.array([True, True])

        result = utils.points_in_a_bounding_box(points, bb_vertices)

        np.testing.assert_array_equal(result, expected)

    def test_points_outside_bounding_box(self):
        bb_vertices = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, -1],
                [1, -1, 1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, -1],
                [-1, -1, 1],
            ]
        )

        points = np.array([[2, 2, 2], [-2, -2, -2]])
        expected = np.array([False, False])

        result = utils.points_in_a_bounding_box(points, bb_vertices)

        np.testing.assert_array_equal(result, expected)


class TestStableUniqueMerge(unittest.TestCase):

    def test_multiple_lists_with_unique_elements(self):
        self.assertEqual(
            utils.stable_unique_merge([1, 2], [3, 4], [5, 6]),
            [1, 2, 3, 4, 5, 6],
        )

    def test_lists_with_overlapping_elements(self):
        self.assertEqual(
            utils.stable_unique_merge([1, 2, 3], [3, 4, 5], [5, 6, 7]),
            [1, 2, 3, 4, 5, 6, 7],
        )

    def test_empty_list_input(self):
        self.assertEqual(utils.stable_unique_merge([], [], []), [])

    def test_no_list_input(self):
        self.assertEqual(utils.stable_unique_merge(), [])

    def test_order_maintenance(self):
        self.assertEqual(
            utils.stable_unique_merge([3, 2, 1], [2, 1, 4]), [3, 2, 1, 4]
        )


class TestMergeTwoArray(unittest.TestCase):

    def test_same_shape(self):
        """Test blending of arrays with the same shape."""
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[4, 3], [2, 1]])
        expected = np.array([[2.5, 2.5], [2.5, 2.5]])
        np.testing.assert_array_almost_equal(
            utils.merge_two_array(arr1, arr2, 0.5), expected
        )

    def test_broadcast_color(self):
        """Test blending an image with a color (broadcasting)."""
        img = np.ones((2, 2, 3))
        color = np.array([0, 0, 1])  # Blue
        expected = np.ones((2, 2, 3)) * 0.5
        expected[:, :, 2] = 1  # Expect blue channel to be dominant
        result = utils.merge_two_array(img, color, 0.5)
        np.testing.assert_array_almost_equal(result, expected)

    def test_incompatible_shapes(self):
        """Test blending with incompatible shapes to raise ValueError."""
        arr1 = np.ones((2, 3))
        arr2 = np.ones((3, 2))
        with self.assertRaises(ValueError):
            utils.merge_two_array(arr1, arr2)


if __name__ == "__main__":
    unittest.main()
