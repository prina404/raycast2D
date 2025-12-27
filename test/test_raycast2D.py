import math
import unittest

import numpy as np

from raycast2D import cast


class TestCastBasicProperties(unittest.TestCase):
    def test_empty_map_single_ray_hits_border_right(self) -> None:
        img = np.full((10, 10), 255, dtype=np.uint8)
        pose = (5, 5, 0.0)  # yaw=0 -> right

        rays = cast(img, pose, num_rays=1, FOV=1, ray_length=100)
        self.assertEqual(rays.shape, (1, 2))

        x, y = map(int, rays[0])
        self.assertEqual((x, y), (img.shape[1] - 1, pose[1]))

    def test_obstacle_above_origin_collision_is_obstacle_cell(self) -> None:
        img = np.full((12, 12), 255, dtype=np.uint8)
        pose = (6, 9, -math.pi / 2)  # up
        img[4, 6] = 0  # obstacle directly above along same column

        rays = cast(img, pose, num_rays=1, FOV=1, ray_length=200)
        x, y = map(int, rays[0])
        self.assertEqual((x, y), (6, 4))

    def test_yaw_controls_direction_right_vs_left(self) -> None:
        img = np.full((11, 11), 255, dtype=np.uint8)
        pose = (5, 5)

        rays_right = cast(img, (pose[0], pose[1], 0.0),
                          num_rays=1, FOV=1, ray_length=200)
        xr, yr = rays_right[0]
        self.assertEqual((xr, yr), (img.shape[1] - 1, pose[1]))

        rays_left = cast(img, (pose[0], pose[1], -math.pi),
                         num_rays=1, FOV=1, ray_length=200)
        xl, yl = rays_left[0]
        self.assertEqual((xl, yl), (0, pose[1]))

    def test_fov_180_default_yaw_first_left_last_right(self) -> None:
        img = np.full((21, 21), 255, dtype=np.uint8)
        pose = (10, 10)  # default yaw=-pi/2

        rays = cast(img, pose, num_rays=2, FOV=180, ray_length=500)
        (x0, y0) = map(int, rays[0])
        (x1, y1) = map(int, rays[-1])

        self.assertEqual((x0, y0), (0, pose[1]))
        self.assertEqual((x1, y1), (img.shape[1] - 1, pose[1]))

    def test_fov_180_yaw0_first_top_last_bottom(self) -> None:
        img = np.full((21, 21), 255, dtype=np.uint8)
        pose = (10, 10, 0.0)

        rays = cast(img, pose, num_rays=2, FOV=180, ray_length=500)
        (x0, y0) = map(int, rays[0])
        (x1, y1) = map(int, rays[-1])

        self.assertEqual((x0, y0), (pose[0], 0))
        self.assertEqual((x1, y1), (pose[0], img.shape[0] - 1))


class TestCastInputValidation(unittest.TestCase):
    def test_pose_on_occupied_cell_raises(self) -> None:
        img = np.full((10, 10), 255, dtype=np.uint8)
        img[3, 4] = 0
        with self.assertRaises(ValueError):
            cast(img, (4, 3))

    def test_rejects_non_integer_dtype(self) -> None:
        img = np.zeros((10, 10), dtype=np.float32)
        img[:] = 1.0
        with self.assertRaises(ValueError):
            cast(img, (5, 5))

    def test_accepts_single_channel_3d_image(self) -> None:
        img = np.full((10, 10, 1), 255, dtype=np.uint8)
        rays = cast(img, (5, 5, 0.0), num_rays=1, FOV=1, ray_length=50)
        self.assertEqual(rays.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
