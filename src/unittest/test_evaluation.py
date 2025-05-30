import os
import sys

os.chdir(os.path.split(os.path.realpath(__file__))[0])
sys.path.append("..")

import unittest

import numpy as np

import evaluation


class TestEvaluation(unittest.TestCase):

    _pred_img = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    _target_img = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
    )

    def test_statistics_from_binary_img_pair(self):

        R, details = evaluation.statistics_from_binary_img_pair(
            self._pred_img, self._target_img, return_details=True
        )

        TP_mask = np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]
        )

        FP_mask = np.array(
            [
                [0, 0, 0],
                [1, 0, 1],
                [0, 0, 0],
            ]
        )

        FN_mask = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )

        TN_mask = np.array(
            [
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1],
            ]
        )

        self.assertEqual(R, {"TP": 2, "FP": 2, "FN": 1, "TN": 4})
        self.assertTrue(np.all(details["TP"] == TP_mask))
        self.assertTrue(np.all(details["FP"] == FP_mask))
        self.assertTrue(np.all(details["FN"] == FN_mask))
        self.assertTrue(np.all(details["TN"] == TN_mask))


if __name__ == "__main__":
    unittest.main()
