import os
import unittest

import src.pcd as pcd


class TestPCD(unittest.TestCase):

    valid_entries = [
        "VERSION",
        "FIELDS",
        "SIZE",
        "TYPE",
        "COUNT",
        "WIDTH",
        "HEIGHT",
        "VIEWPOINT",
        "POINTS",
        "DATA",
    ]

    _path = os.path.dirname(os.path.realpath(__file__))
    _path = os.path.join(_path, "example.pcd")

    def test_read(self):
        foo = pcd.read(self._path)

        self.assertTrue(foo.shape == (213,))
        self.assertTrue(foo.dtype.names == ("x", "y", "z", "rgb"))
        self.assertTrue(foo.dtype.itemsize == 16)

    def test_read_header(self):

        _, header = pcd.read(self._path, return_header=True)

        for entry in self.valid_entries:
            self.assertTrue(entry in header)

    def test_write(self):

        foo = pcd.read(self._path)

        output_path = self._path.replace("example.pcd", "output.pcd")
        pcd.write(output_path, foo)

        bar = pcd.read(output_path)

        self.assertTrue(foo.shape == bar.shape)
        self.assertTrue(foo.dtype.names == bar.dtype.names)
        self.assertTrue(foo.dtype.itemsize == bar.dtype.itemsize)

        os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
