import math

import numpy as np
import pytest

from raycast2D import cast


def test_empty_map_single_ray_hits_border_right() -> None:
    img = np.full((10, 10), 255, dtype=np.uint8)
    pose = (5, 5, 0.0)  # yaw=0 -> right

    rays = cast(img, pose, num_rays=1, FOV=1, ray_length=100)
    assert rays.shape == (1, 2)

    x, y = map(int, rays[0])
    assert (x, y) == (img.shape[1] - 1, pose[1])


def test_obstacle_above_origin_collision_is_obstacle_cell() -> None:
    img = np.full((12, 12), 255, dtype=np.uint8)
    pose = (6, 9, -math.pi / 2)  # up
    img[4, 6] = 0  # obstacle directly above along same column

    rays = cast(img, pose, num_rays=1, FOV=1, ray_length=200)
    x, y = map(int, rays[0])
    assert (x, y) == (6, 4)


def test_yaw_controls_direction_right_vs_left() -> None:
    img = np.full((11, 11), 255, dtype=np.uint8)
    pose = (5, 5)

    rays_right = cast(img, (pose[0], pose[1], 0.0),
                      num_rays=1, FOV=1, ray_length=200)
    xr, yr = rays_right[0]
    assert (xr, yr) == (img.shape[1] - 1, pose[1])

    rays_left = cast(img, (pose[0], pose[1], -math.pi),
                     num_rays=1, FOV=1, ray_length=200)
    xl, yl = rays_left[0]
    assert (xl, yl) == (0, pose[1])


def test_fov_180_default_yaw_first_left_last_right() -> None:
    img = np.full((21, 21), 255, dtype=np.uint8)
    pose = (10, 10)  # default yaw=-pi/2

    rays = cast(img, pose, num_rays=2, FOV=180, ray_length=500)
    (x0, y0) = map(int, rays[0])
    (x1, y1) = map(int, rays[-1])

    assert (x0, y0) == (0, pose[1])
    assert (x1, y1) == (img.shape[1] - 1, pose[1])


def test_fov_180_yaw0_first_top_last_bottom() -> None:
    img = np.full((21, 21), 255, dtype=np.uint8)
    pose = (10, 10, 0.0)

    rays = cast(img, pose, num_rays=2, FOV=180, ray_length=500)
    (x0, y0) = map(int, rays[0])
    (x1, y1) = map(int, rays[-1])

    assert (x0, y0) == (pose[0], 0)
    assert (x1, y1) == (pose[0], img.shape[0] - 1)


def test_pose_on_occupied_cell_raises() -> None:
    img = np.full((10, 10), 255, dtype=np.uint8)
    img[3, 4] = 0
    with pytest.raises(ValueError):
        cast(img, (4, 3))


def test_rejects_non_integer_dtype() -> None:
    img = np.zeros((10, 10), dtype=np.float32)
    img[:] = 1.0
    with pytest.raises(ValueError):
        cast(img, (5, 5))


def test_accepts_single_channel_3d_image() -> None:
    img = np.full((10, 10, 1), 255, dtype=np.uint8)
    rays = cast(img, (5, 5, 0.0), num_rays=1, FOV=1, ray_length=50)
    assert rays.shape == (1, 2)


def test_only_true_collisions_true_returns_empty_when_no_collision() -> None:
    img = np.full((30, 30), 255, dtype=np.uint8)
    pose = (15, 15, 0.0)

    # Small ray_length keeps endpoints inside free space (no border hit, no obstacles)
    rays_true = cast(
        img,
        pose,
        num_rays=5,
        FOV=30,
        ray_length=3,
        only_true_collisions=True,
    )
    assert rays_true.shape == (0, 2)

    rays_false = cast(
        img,
        pose,
        num_rays=5,
        FOV=30,
        ray_length=3,
        only_true_collisions=False,
    )
    assert rays_false.shape == (5, 2)


def test_only_true_collisions_filters_to_obstacle_hits_subset() -> None:
    img = np.full((30, 30), 255, dtype=np.uint8)
    pose = (15, 15, 0.0)

    # With num_rays=2 and FOV=180 around yaw=0, angles are exactly -pi/2 (up) and +pi/2 (down)
    # Put an obstacle only on the "up" ray.
    img[13, 15] = 0  # (x=15, y=13)

    rays_true = cast(
        img,
        pose,
        num_rays=2,
        FOV=180,
        ray_length=3,
        only_true_collisions=True,
    )
    assert rays_true.shape == (1, 2)
    assert tuple(map(int, rays_true[0])) == (15, 13)

    rays_false = cast(
        img,
        pose,
        num_rays=2,
        FOV=180,
        ray_length=3,
        only_true_collisions=False,
    )
    assert rays_false.shape == (2, 2)

    pts = {tuple(map(int, p)) for p in rays_false}
    # One ray hits the obstacle; the other ends in free space (no collision)
    assert (15, 13) in pts
    assert (15, 18) in pts
