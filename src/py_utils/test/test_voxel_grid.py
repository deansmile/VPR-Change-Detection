import unittest

import numpy as np

import src.voxel_grid as voxel_grid


class TestVoxelGrid(unittest.TestCase):

    _points = np.array(
        [
            [0.1, 0.1, 0.1],  # <- 1st voxel
            [0.4, 0.4, 0.4],  # <- 1st voxel
            [0.8, 0.8, 0.8],  # <- 2nd voxel
            [1.2, 1.2, 1.2],  # <- 3rd voxel
            [1.6, 1.6, 1.6],  # <- 4th voxel
            [2.0, 2.0, 2.0],  # <- 5th voxel
        ]
    )

    # fmt: off
    _attributes = [
        np.r_[1, 2, 3, 4, 5, 6],
        np.array(
            [
                [0.5, 0.5],
                [0.3, 0.3],
                [0.8, 0.8],
                [1.2, 1.2],
                [1.6, 1.6],
                [2.0, 2.0],
            ]
        ),
        np.array(
            [
                [0.5, 0.5, 0.5],
                [0.3, 0.3, 0.3],
                [0.8, 0.8, 0.8],
                [1.2, 1.2, 1.2],
                [1.6, 1.6, 1.6],
                [2.0, 2.0, 2.0],
            ]
        ),
    ]
    # fmt: on

    _voxel_size = 0.5

    def test_voxel_grid_length(self):

        vg = voxel_grid.VoxelGrid(
            self._points,
            voxel_size=self._voxel_size,
            attributes=self._attributes,
        )

        # Check number of voxels
        self.assertEqual(len(vg), 5)

    def test_voxel_centers(self):

        vg = voxel_grid.VoxelGrid(
            self._points,
            voxel_size=self._voxel_size,
            attributes=self._attributes,
        )

        # fmt: off
        # Expected voxel indices
        expected_indices = np.array([
            [ .25,  .25,  .25],
            [ .75,  .75,  .75],
            [1.25, 1.25, 1.25],
            [1.75, 1.75, 1.75],
            [2.25, 2.25, 2.25],
        ])
        # fmt: on

        self.assertTrue(np.all(vg.voxel_centers == expected_indices))

    def test_voxel_centroids(self):

        vg = voxel_grid.VoxelGrid(
            self._points,
            voxel_size=self._voxel_size,
            attributes=self._attributes,
        )

        # fmt: off
        expected_centroids = np.array(
            [
                [0.25, 0.25, 0.25],
                [ 0.8,  0.8,  0.8],
                [ 1.2,  1.2,  1.2],
                [ 1.6,  1.6,  1.6],
                [ 2.0,  2.0,  2.0],
            ]
        )
        # fmt: on

        self.assertTrue(np.allclose(vg.voxel_centroids, expected_centroids))

    def test_voxel_attributes(self):

        vg = voxel_grid.VoxelGrid(
            self._points,
            voxel_size=self._voxel_size,
            attributes=self._attributes,
        )

        # fmt: off
        expected_attributes = [
            np.r_[1.5, 3, 4, 5, 6],
            np.array(
                [
                    [0.4, 0.4],
                    [0.8, 0.8],
                    [1.2, 1.2],
                    [1.6, 1.6],
                    [2.0, 2.0],
                ]
            ),
            np.array(
                [
                    [0.4, 0.4, 0.4],
                    [0.8, 0.8, 0.8],
                    [1.2, 1.2, 1.2],
                    [1.6, 1.6, 1.6],
                    [2.0, 2.0, 2.0],
                ]
            ),
        ]
        # fmt: on

        attributes = vg.voxel_attributes

        self.assertTrue(np.allclose(attributes[0], expected_attributes[0]))
        self.assertTrue(np.allclose(attributes[1], expected_attributes[1]))
        self.assertTrue(np.allclose(attributes[2], expected_attributes[2]))

    def test_voxel_counts(self):

        vg = voxel_grid.VoxelGrid(
            self._points,
            voxel_size=self._voxel_size,
            attributes=self._attributes,
        )

        expected_counts = np.array([2, 1, 1, 1, 1])
        self.assertTrue(np.all(vg.voxel_counts == expected_counts))


if __name__ == "__main__":
    unittest.main()
