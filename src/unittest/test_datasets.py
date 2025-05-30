import os
import sys

os.chdir(os.path.split(os.path.realpath(__file__))[0])
sys.path.append("..")

import unittest

import numpy as np

import datasets


class TestDataLoader(unittest.TestCase):

    def test_list_path(self):

        expected_keys = ["VL_CMU_CD", "PSCD"]

        for key in expected_keys:
            msg = "fail on %s" % key

            self.assertTrue(key in datasets.list_data(), msg=msg)
            self.assertTrue(key in datasets.list_path(), msg=msg)

    def test_VL_CMU_CD(self):

        name = "VL_CMU_CD"

        trainset = datasets.get_dataset(name, mode="train")
        testset = datasets.get_dataset(name, mode="test")
        valset = datasets.get_dataset(name, mode="val")

        self.assertEqual(len(trainset) + len(valset), 3732)
        self.assertEqual(len(testset), 429)

        for dataset in [trainset, testset, valset]:

            for n, (t0, t1, gt) in enumerate(dataset):

                self.assertEqual(t0.shape, t1.shape)
                self.assertEqual(t0.shape[:2], gt.shape)
                self.assertEqual(t0.shape, (512, 512, 3))
                self.assertTrue(np.all(t0 <= 1.0))
                self.assertTrue(np.all(t0 >= 0.0))

                if n >= 10:
                    break

    def test_PSCD(self):

        name = "PSCD"

        trainset = datasets.get_dataset(name, mode="train")
        testset = datasets.get_dataset(name, mode="test")
        valset = datasets.get_dataset(name, mode="val")

        self.assertEqual(len(trainset), 9240)
        self.assertEqual(len(valset), 1155)
        self.assertEqual(len(testset), 1155)

        for dataset in [trainset, testset, valset]:

            for n, (t0, t1, gt) in enumerate(dataset):

                self.assertEqual(t0.shape, t1.shape)
                self.assertEqual(t0.shape[:2], gt.shape)
                self.assertEqual(t0.shape, (224, 224, 3))
                self.assertTrue(np.all(t0 <= 1.0))
                self.assertTrue(np.all(t0 >= 0.0))

                if n >= 10:
                    break

    def test_PSCD_Diff_View_1(self):

        name = "PSCD_Diff_View"

        opts = {"adjacent_distance": 1}

        trainset = datasets.get_dataset(name, mode="train", **opts)
        testset = datasets.get_dataset(name, mode="test", **opts)
        valset = datasets.get_dataset(name, mode="val", **opts)

        self.assertEqual(len(trainset), 9240 / 15 * 14)
        self.assertEqual(len(valset), 1155 / 15 * 14)
        self.assertEqual(len(testset), 1155 / 15 * 14)

        for dataset in [trainset, testset, valset]:

            for n, (t0, t1, gt) in enumerate(dataset):

                self.assertEqual(t0.shape, t1.shape)
                self.assertEqual(t0.shape[:2], gt.shape)
                self.assertEqual(t0.shape, (224, 224, 3))
                self.assertTrue(np.all(t0 <= 1.0))
                self.assertTrue(np.all(t0 >= 0.0))

                if n >= 10:
                    break

    def test_PSCD_Diff_View_2(self):

        name = "PSCD_Diff_View"

        opts = {"adjacent_distance": 2}

        trainset = datasets.get_dataset(name, mode="train", **opts)
        testset = datasets.get_dataset(name, mode="test", **opts)
        valset = datasets.get_dataset(name, mode="val", **opts)

        self.assertEqual(len(trainset), 9240 / 15 * 13)
        self.assertEqual(len(valset), 1155 / 15 * 13)
        self.assertEqual(len(testset), 1155 / 15 * 13)

        for dataset in [trainset, testset, valset]:

            for n, (t0, t1, gt) in enumerate(dataset):

                self.assertEqual(t0.shape, t1.shape)
                self.assertEqual(t0.shape[:2], gt.shape)
                self.assertEqual(t0.shape, (224, 224, 3))
                self.assertTrue(np.all(t0 <= 1.0))
                self.assertTrue(np.all(t0 >= 0.0))

                if n >= 10:
                    break


if __name__ == "__main__":
    unittest.main()
