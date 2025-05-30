import numpy as np
from . import utils_segmentation


def _is_n_by_3_array(x):

    if not isinstance(x, np.ndarray):
        return False

    if len(x.shape) != 2:
        return False

    if x.shape[1] != 3:
        return False

    return True


def _sort_for_voxel_grid(points, voxel_size):

    # get the voxel index
    voxel_index = np.floor(points / voxel_size).astype(np.int64)

    # sort the points (x -> y -> z)
    sort_index = np.lexsort(
        (voxel_index[:, 2], voxel_index[:, 1], voxel_index[:, 0])
    )
    points = points[sort_index]
    voxel_index = voxel_index[sort_index]

    return points, voxel_index, sort_index


class VoxelGrid:
    """
    VoxelGrid class for voxelizing 3D points.

    There are different ways to define the location of a voxel,
    such as using the center or the vertices as the reference point.
    In this implementation, the origin of the grid is aligned with
    the vertices of the voxels, rather than their centers.

                *----*----*----*
               /    /    /    /|
              /    /    /    / *
             /    /    /    / /|
            *----*----*----* / *
                 |    |    |/ /
    (0, 0, 0) -> *----*----* /
                 |    |    |/
                 *----*----*

    """

    def __init__(self, points, voxel_size=[0.5, 0.5, 0.5], attributes=[]):

        # Check `points`
        if not _is_n_by_3_array(points):
            raise ValueError("Points must an array in (n, 3) shape.")

        if not np.iterable(voxel_size):
            voxel_size = [voxel_size] * 3
        voxel_size = np.abs(np.array(voxel_size))

        # Check `attributes`
        for n, attr in enumerate(attributes):
            if len(attr) == len(points):
                continue
            msg = f"{n}th attribute must have the same length as points."
            raise ValueError(msg)

        # store the input
        X = _sort_for_voxel_grid(points, voxel_size)
        points, voxel_index, sort_index = X

        splits = np.any(voxel_index[1:] != voxel_index[:-1], axis=-1)
        splits = np.nonzero(splits)[0] + 1
        splits = np.r_[0, splits, len(voxel_index)]

        self._sorted_points = points
        self._sorted_voxel_index = voxel_index
        self._to_point_indices = sort_index
        self._voxel_size = voxel_size
        self._attributes = [i[sort_index] for i in attributes]
        self._splits = splits

    def __len__(self):
        return len(self._splits) - 1

    @property
    def sorted_original_points(self):
        return self._sorted_points

    @property
    def sorted_original_attributes(self):
        return self._attributes

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def original_to_sorted_point_indices(self):
        return self._to_point_indices

    @property
    def sorted_to_original_point_indices(self):
        return np.argsort(self._to_point_indices)

    # === voxelize output ===
    @property
    def voxel_indices(self):
        return self._sorted_voxel_index[self._splits[:-1]]

    @property
    def voxel_centers(self):
        voxel_index = self.voxel_indices
        return voxel_index * self._voxel_size + self._voxel_size / 2

    @property
    def voxel_centroids(self):

        if hasattr(self, "_voxel_centroids"):
            return self._voxel_centroids

        # compute the mean points for each voxel
        centroids = utils_segmentation.segmented_mean(
            self._sorted_points,
            s_ind=self._splits[:-1],  # start_ind
            e_ind=self._splits[1:],  # end_ind
        )
        self._voxel_centroids = centroids
        return self._voxel_centroids

    @property
    def voxel_attributes(self):

        if hasattr(self, "_voxel_attributes"):
            return self._voxel_attributes

        # compute the mean attributes for each voxel
        voxel_attributes = []
        for attr in self._attributes:
            voxel_attr = utils_segmentation.segmented_mean(
                attr,
                s_ind=self._splits[:-1],  # start_ind
                e_ind=self._splits[1:],  # end_ind
            )
            voxel_attributes.append(voxel_attr)

        self._voxel_attributes = voxel_attributes
        return self._voxel_attributes

    @property
    def voxel_counts(self):
        return np.diff(self._splits)
