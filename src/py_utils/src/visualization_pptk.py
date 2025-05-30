import scipy.interpolate
import numpy as np


def make_coordinates(xyz, R, length=1.0, number=20):
    # TODO: array shape check consistence

    steps = np.linspace(0, length, number)  # (number, )

    R = R.transpose([0, 2, 1])

    N = len(xyz)

    # (N, 1, 1, 3) + (N, 3, 1, 3) * (number, 1) = (N, 3, number, 3)
    points = xyz[..., None, None, :] + R[..., None, :] * steps[:, None]

    colors = np.array(
        [
            [1.0, 0.0, 0.0],  # red
            [0.0, 1.0, 0.0],  # green
            [0.0, 0.0, 1.0],  # blue
        ]
    )

    # (N, 3, number, 1) * (1, 3, 1, 3)
    colors = np.ones((N, 3, number, 1)) * colors[:, None, :]

    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    return points, colors


def make_lines(src, dst, start=True, end=True, eps=0.005, return_splits=False):
    """
    TODO add docstring and comments
    """

    V = dst - src
    L = np.sqrt(np.sum(V**2, axis=-1))

    num = np.maximum(2, np.floor(L / eps).astype(np.int64)) + 1
    splits = np.r_[0, np.cumsum(num)]

    xyz = np.empty((splits[-1], 3), dtype=np.float64)
    for i, j, n, s in zip(src, dst, num, splits[:-1]):
        xyz[s : s + n] = np.linspace(i, j, n)

    I = np.ones(splits[-1], dtype=np.bool_)

    if not start:
        I[splits[:-1]] = 0
        num = num - 1

    if not end:
        I[splits[1:] - 1] = 0
        num = num - 1

    if return_splits:
        return xyz[I], np.r_[0, np.cumsum(num)]
    return xyz[I]


def make_polygon(xyz, eps=0.005):
    A = xyz
    B = np.vstack([xyz[1:], xyz[:1]])
    L = make_lines(A, B, start=True, end=False, eps=eps)
    return L


def make_polyline(xyz, eps=0.005):
    L = make_lines(xyz[:-1], xyz[1:], start=True, end=False, eps=eps)
    return np.vstack([L, xyz[-1:]])


def make_bounding_box_vertices(lxs, lys, lzs):
    """
    Notes:
    ------
        4----0
       /|   /|
      / 5--/-1
     / /  / /
    7----3 /  z y
    |/   |/   |/
    6----2    .--x
    """

    shp = np.shape(lxs)
    vertices = np.empty(shp + (8, 3), dtype=np.float64)

    vertices[..., 0:4, 0] = lxs[:, None]
    vertices[..., 4:8, 0] = -lxs[:, None]
    vertices[..., [0, 1, 4, 5], 1] = lys[:, None]
    vertices[..., [2, 3, 6, 7], 1] = -lys[:, None]
    vertices[..., [0, 3, 4, 7], 2] = lzs[:, None]
    vertices[..., [1, 2, 5, 6], 2] = -lzs[:, None]

    vertices = vertices / 2.0
    return vertices


def _make_bounding_box_lines(vertices, eps=0.005):
    """
    xyz: (N, 3)
    lx:  (N,)
    ly:  (N,)
    lz:  (N,)
    """

    assert np.shape(vertices) == (8, 3)

    I = [0, 1, 5, 4]
    J = [3, 2, 6, 7]
    points = np.vstack(
        [
            make_polygon(vertices[I], eps=eps),
            make_polygon(vertices[J], eps=eps),
            make_lines(
                vertices[I], vertices[J], start=False, end=False, eps=eps
            ),
        ]
    )
    return points


def make_bounding_boxes_lines(vertices, eps=0.005):

    assert np.shape(vertices)[-2:] == (8, 3)

    if len(np.shape(vertices)) == 2:
        points = _make_bounding_box_lines(vertices, eps=eps)
        return [points]

    points = []
    for vertex in vertices:
        points.append(_make_bounding_box_lines(vertex, eps=eps))
    return points


def make_color(s_min, s_max, color_map=None):
    """
    TODO: add docstring
    """

    # default color map(jet)
    _colormap = np.array(
        [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    )

    if color_map is None:
        color_map = _colormap

    if s_min == s_max:
        return lambda x: color_map[0]

    def wrap(x):
        foo = scipy.interpolate.interp1d(
            np.linspace(s_min, s_max, len(color_map)), color_map, axis=0
        )

        x = np.minimum(x, s_max)
        x = np.maximum(x, s_min)
        return foo(x)

    return wrap
