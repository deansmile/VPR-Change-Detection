import os
import unittest
import pickle

import numpy as np

import src.array_data as array_data

_path = os.path.dirname(os.path.realpath(__file__))


class TestArray(unittest.TestCase):

    _data = array_data.Array(10)

    def test_array(self):
        self.assertEqual(len(self._data), 10)
        self.assertTrue(np.all(self._data.index == np.arange(10)))

    def test_getitem(self):

        # integer indexing
        sub_data = self._data[3]
        self.assertEqual(len(sub_data), 1)
        self.assertTrue(np.all(sub_data.index == [3]))

        # slice indexing
        sub_data = self._data[3:5]
        self.assertEqual(len(sub_data), 2)
        self.assertTrue(np.all(sub_data.index == [3, 4]))

        # boolean indexing
        mask = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        sub_data = self._data[mask]
        self.assertEqual(len(sub_data), 2)
        self.assertTrue(np.all(sub_data.index == [0, 2]))

        # list indexing (unique)
        sub_data = self._data[[3, 5, 7]]
        self.assertEqual(len(sub_data), 3)
        self.assertTrue(np.all(sub_data.index == [3, 5, 7]))

        # list indexing (repeated)
        sub_data = self._data[[7, 5, 3, 3, 3]]
        self.assertEqual(len(sub_data), 3)
        self.assertTrue(np.all(sub_data.index == [3, 5, 7]))

        # numpy indexing (unique)
        sub_data = self._data[np.array([3, 5, 7])]
        self.assertEqual(len(sub_data), 3)
        self.assertTrue(np.all(sub_data.index == [3, 5, 7]))

        # numpy indexing (repeated)
        sub_data = self._data[np.array([7, 5, 3, 3, 3])]
        self.assertEqual(len(sub_data), 3)
        self.assertTrue(np.all(sub_data.index == [3, 5, 7]))

    def test_pickle(self):

        with open(os.path.join(_path, "test.pkl"), "wb") as f:
            pickle.dump(self._data, f)

        with open(os.path.join(_path, "test.pkl"), "rb") as f:
            foo = pickle.load(f)

        self.assertEqual(len(foo), len(self._data))
        self.assertTrue(np.all(foo.index == self._data.index))

        os.remove(os.path.join(_path, "test.pkl"))


if __name__ == "__main__":
    unittest.main()
