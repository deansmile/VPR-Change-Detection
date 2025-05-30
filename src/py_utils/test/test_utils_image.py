import unittest

import numpy as np

import src.utils_img as utils_img


class TestImageOverlay(unittest.TestCase):

    def test_simple_overlay(self):
        """Test simple overlay without a mask."""
        img = np.zeros((2, 2, 3))
        layer = np.ones((2, 2, 3))
        expected = np.ones((2, 2, 3)) * 0.5
        result = utils_img.overlay_image(img, layer, 0.5)
        np.testing.assert_array_almost_equal(result, expected)

    def test_clamping(self):
        """Test the clamping function on out-of-range values."""
        img = np.array([[-0.1, 1.1], [0.5, 0]])
        expected = np.array([[0, 1], [0.5, 0]])
        result = utils_img.clamp_image(img)
        np.testing.assert_array_almost_equal(result, expected)

    def test_overlay_with_mask(self):
        """Test overlay with a mask."""
        img = np.zeros((2, 2, 3))
        layer = np.ones((2, 2, 3))
        mask = np.array([[True, False], [False, True]])
        expected = img.copy()
        expected[mask] = 0.5
        result = utils_img.overlay_image(img, layer, 0.5, mask)
        np.testing.assert_array_almost_equal(result, expected)

    def test_overlay_with_color_layer(self):
        """Test overlay with a color layer."""
        img = np.zeros((2, 2, 3))
        color_layer = np.array([1, 0, 0])  # Red
        expected = np.zeros((2, 2, 3))
        expected[:, :, 0] = 0.5  # Blend red channel
        result = utils_img.overlay_image(img, color_layer, 0.5)
        np.testing.assert_array_almost_equal(result, expected)

    def test_incompatible_mask_shape(self):
        """Test overlay with an incompatible mask shape."""
        img = np.zeros((2, 2, 3))
        layer = np.ones((2, 2, 3))
        mask = np.array([True, False, False])  # Incompatible mask
        with self.assertRaises(ValueError):
            utils_img.overlay_image(img, layer, 0.5, mask)


if __name__ == "__main__":
    unittest.main()
