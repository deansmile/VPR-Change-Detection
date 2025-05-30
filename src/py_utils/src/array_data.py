import copy

import numpy as np
import pandas as pd

from . import utils


class Array:
    _columns = ["index"]
    _dtypes = [np.int64]

    def __init__(self, n):
        self._allocate(n)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return "<Array contains %d rows>" % len(self._data)

    def __getitem__(self, key):

        # fmt: off
        if (
            not isinstance(key, slice)
            and not (isinstance(key, np.ndarray) and key.dtype == np.bool_)
        ):
            key = np.unique(key)
        # fmt: on

        other = copy.copy(self)
        other._data = self._data.iloc[key]
        return other

    def __getstate__(self):
        state = {i: self._data[i].to_numpy() for i in self._columns}
        return state

    def __setstate__(self, state):
        n = len(state["index"])
        self._allocate(n)
        self._data.index = state["index"]

        for i in self._columns:
            self._data[i] = state[i]

    def _allocate(self, n):
        dt = np.dtype(list(zip(self._columns, self._dtypes)))
        self._data = pd.DataFrame(np.empty(n, dtype=dt))
        self._data["index"] = self._data.index

    @property
    def index(self):
        return self._data["index"].to_numpy()


class Timestamps(Array):
    _columns = Array._columns + ["timestamp"]
    _dtypes = Array._dtypes + [np.int64]

    def __init__(self, n):
        super().__init__(n)

    def __repr__(self):
        return "<Timestamps contains %d rows>" % len(self._data)

    @property
    def datetime(self):
        return pd.to_datetime(self.timestamps)

    @property
    def timestamps(self):
        return self._data["timestamp"].to_numpy()

    @timestamps.setter
    def timestamps(self, value):
        self._data["timestamp"] = np.array(value, dtype=np.int64)


##############
# Trajectory #
##############


class PoseSequence(Array):

    _columns = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    _dtypes = [np.float64] * len(_columns)

    _columns = Array._columns + _columns
    _dtypes = Array._dtypes + _dtypes

    def __init__(self, n):
        super().__init__(n)

    def __repr__(self):
        return "<PoseSequence contains %d rows>" % len(self._data)

    def adjust(self, adjustment):
        raise NotImplementedError

    @property
    def xyz(self):
        return self._data[["x", "y", "z"]].to_numpy()

    @property
    def quaternion(self):
        return self._data[["qx", "qy", "qz", "qw"]].to_numpy()

    @property
    def R(self):
        return utils.Q_to_R(self.quaternion)

    @xyz.setter
    def xyz(self, value):
        self._data[["x", "y", "z"]] = value

    @quaternion.setter
    def quaternion(self, value):
        self._data[["qx", "qy", "qz", "qw"]] = value


class TimePoseSequence(Timestamps, PoseSequence):

    # exclude 'index' field from PoseSequence
    # because Timestamps has the same field
    _columns = Timestamps._columns + PoseSequence._columns[1:]
    _dtypes = Timestamps._dtypes + PoseSequence._dtypes[1:]

    def __init__(self, n):
        super().__init__(n)

    def __repr__(self):
        return "<TimePoseSequence contains %d rows>" % len(self._data)
