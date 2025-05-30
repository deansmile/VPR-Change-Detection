import datetime
import os
import sys

import numpy as np
import requests


def is_connect_to_network(url="https://www.google.com", timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200

    except requests.RequestException:
        return False


def is_in_notebook():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except:
        return False


def get_utc_time(format="%Y-%m-%d.%H-%M-%S"):
    utctime = datetime.datetime.now(datetime.timezone.utc)
    utctime = utctime.strftime(format)
    return utctime


def mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


def mkpath(path, root):
    paths = path.split("/")
    for i in paths:
        root = os.path.join(root, i)
        mkdir(root)


def find_directory_by_DFS(root, condition):

    assert callable(condition)

    R = []

    def f(path):
        if condition(path):
            R.append(path)

        if not os.path.isdir(path):
            return

        for i in os.listdir(path):
            f(os.path.join(path, i))

    f(root)
    return sorted(R)


class ProgressTimer:

    def __init__(self, bar_length=30, prefix="", verbose=True):

        # timer
        self.start_time = None
        self.end_time = None

        # progress
        self.items = 0
        self.prefix = prefix
        self.verbose = verbose
        self.total_items = None
        self.bar_length = bar_length

        # environment
        self._in_notebook = is_in_notebook()

    def _xprint(self, s):
        if not self.verbose:
            return

        sys.stdout.write(s)
        sys.stdout.flush()

    def _progress_bar(self, first_call=True):
        if not first_call:
            # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
            # \033[1A: move cursor up 1 line
            # \033[2K: erase the entire line
            # \r:      move cursor to the start of line
            msg = "\r" if self._in_notebook else "\033[1A\033[2K\r"
            self._xprint(msg)

        percent = 100.0 * self.items / self.total_items
        percent = round(percent, 1)
        percent = min(max(percent, 0.0), 100.0)

        n1 = int(percent / 100 * self.bar_length)
        n2 = self.bar_length - n1

        msg = self.prefix
        msg += "[" + "=" * n1 + "-" * n2 + "]"
        msg += f" {percent}%"

        postfix = "" if self._in_notebook else "\n"
        if self._in_notebook and percent >= 100:
            postfix = "\n"

        msg = msg + postfix
        self._xprint(msg)

    def tic(self, total_items):
        self.items = 0
        self.total_items = total_items
        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        self.end_time = datetime.datetime.now(datetime.timezone.utc)
        self._progress_bar(first_call=True)

    def toc(self, add=1):
        self.end_time = datetime.datetime.now(datetime.timezone.utc)
        self.items += add
        self._progress_bar(first_call=False)

    @property
    def total_seconds(self):
        return (self.end_time - self.start_time).total_seconds()


def Q_to_R(q):
    if np.shape(q)[-1] != 4:
        raise ValueError("shape of q must be (..., 4)")

    # Normalize quaternions
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.empty(np.shape(q)[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * y**2 - 2 * z**2
    R[..., 0, 1] = 2 * x * y - 2 * w * z
    R[..., 0, 2] = 2 * x * z + 2 * w * y
    R[..., 1, 0] = 2 * x * y + 2 * w * z
    R[..., 1, 1] = 1 - 2 * x**2 - 2 * z**2
    R[..., 1, 2] = 2 * y * z - 2 * w * x
    R[..., 2, 0] = 2 * x * z - 2 * w * y
    R[..., 2, 1] = 2 * y * z + 2 * w * x
    R[..., 2, 2] = 1 - 2 * x**2 - 2 * y**2
    return R


def euler_to_R(euler):

    if np.shape(euler)[-1] != 3:
        raise ValueError("shape of euler must be (..., 3)")

    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # https://en.wikipedia.org/wiki/Rotation_matrix -> General 3D rotations
    R = np.empty(np.shape(euler)[:-1] + (3, 3))
    R[..., 0, 0] = cp * cy
    R[..., 0, 1] = cp * sy
    R[..., 0, 2] = -sp
    R[..., 1, 0] = sr * sp * cy - cr * sy
    R[..., 1, 1] = sr * sp * sy + cr * cy
    R[..., 1, 2] = sr * cp
    R[..., 2, 0] = cr * sp * cy + sr * sy
    R[..., 2, 1] = cr * sp * sy - sr * cy
    R[..., 2, 2] = cr * cp
    return R


def combine_two_mean_var(cnt_mean_var_1, cnt_mean_var_2):
    """
    TODO add docstring
    """
    cnt_1, mean_1, var_1 = cnt_mean_var_1
    cnt_2, mean_2, var_2 = cnt_mean_var_2

    out_cnt = cnt_1 + cnt_2

    if out_cnt == 0:
        return 0, 0, 0

    out_mean = mean_1 * cnt_1 + mean_2 * cnt_2
    out_mean = out_mean / out_cnt

    diff_1 = (mean_1 - out_mean) ** 2
    diff_2 = (mean_2 - out_mean) ** 2

    out_var = cnt_1 * (diff_1 + var_1) + cnt_2 * (diff_2 + var_2)
    out_var = out_var / out_cnt

    return out_cnt, out_mean, out_var


def combine_mean_var(*args):
    """
    Parameters
    ----------
    TODO add docstring
    """
    if len(args) == 0:
        return [0, 0, 0]

    result = args[0]
    for arg in args[1:]:
        result = combine_two_mean_var(result, arg)
    return result


def stable_unique_merge(*lists):

    if len(lists) < 1:
        return []

    def f(l1, l2):
        for i in l2:
            if i in l1:
                continue
            l1.append(i)
        return l1

    L = lists[0]
    for l in lists[1:]:
        L = f(L, l)
    return L


def points_in_a_bounding_box(points, bb_vertices):
    """
    Assumptions:

    bb_vertices

        4----0
       /|   /|
      / 5--/-1
     / /  / /
    7----3 /  z y
    |/   |/   |/
    6----2    .--x
    """

    v0 = points - bb_vertices[0]

    v1 = bb_vertices[3] - bb_vertices[0]
    v2 = bb_vertices[4] - bb_vertices[0]
    v3 = bb_vertices[1] - bb_vertices[0]

    d1 = np.sqrt(np.sum(v1**2))
    d2 = np.sqrt(np.sum(v2**2))
    d3 = np.sqrt(np.sum(v3**2))

    v01 = np.sum(v0 * v1 / d1, axis=-1)
    v02 = np.sum(v0 * v2 / d2, axis=-1)
    v03 = np.sum(v0 * v3 / d3, axis=-1)

    mask = (
        (v01 >= 0)
        & (v01 <= d1)
        & (v02 >= 0)
        & (v02 <= d2)
        & (v03 >= 0)
        & (v03 <= d3)
    )

    return mask


def merge_two_array(array_1, array_2, ratio=0.5):
    """
    Merge two arrays with a specified ratio, supporting broadcasting
    for different shapes.

    Parameters:
    - array_1: The first input array. Can be a list or a numpy array.
    - array_2: The second input array. Can be a list or a numpy array.
    - ratio  : A float between 0 and 1 that determines the weight of
               array_1 in the blend. The default value is 0.5.

    Returns:
    - A numpy array containing the blended result.

    Note:
    - The blending is performed in a linear fashion:
      result = array_1 * ratio + array_2 * (1 - ratio).
    """

    array_1 = np.array(array_1)
    array_2 = np.array(array_2)

    shp_1 = array_1.shape
    shp_2 = array_2.shape
    if shp_1 == shp_2:
        return array_1 * ratio + array_2 * (1 - ratio)

    n = len(shp_1)
    m = len(shp_2)
    target_shp = shp_1 if n > m else shp_2

    if target_shp[-n:] != shp_1 or target_shp[-m:] != shp_2:
        raise ValueError(
            f"Cannot handle arrays merge with {shp_1} and {shp_2}"
        )

    array_1 = np.broadcast_to(array_1, target_shp)
    array_2 = np.broadcast_to(array_2, target_shp)

    return array_1 * ratio + array_2 * (1 - ratio)


def randomize_labels(labels):
    """
    Randomize the order of the labels.

    Parameters:
    - labels: A numpy array of labels.

    Returns:
    - A numpy array of randomized labels.
    """

    labels = np.array(labels)
    unique_labels = np.unique(labels)
    np.random.shuffle(unique_labels)

    label_map = {l: n for n, l in enumerate(unique_labels)}
    return np.array([label_map[i] for i in labels])
