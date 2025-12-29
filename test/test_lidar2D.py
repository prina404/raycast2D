import math
import warnings

import numpy as np
import pytest

from raycast2D.raycast2D import Lidar2D
import raycast2D.raycast2D as rc_mod


def _scan(
    img: np.ndarray,
    pose: tuple[int, int] | tuple[int, int, float],
    *,
    num_rays: int = 1000,
    FOV: int = 360,
    ray_length: int = 500,
    only_true_collisions: bool = True,
):
    lidar = Lidar2D(num_rays=num_rays, FOV=FOV, ray_length=ray_length)
    return lidar.scan(pose, image=img, only_true_collisions=only_true_collisions)


def test_empty_map_single_ray_hits_border_right() -> None:
    img = np.full((10, 10), 255, dtype=np.uint8)
    pose = (5, 5, math.pi / 2)  # yaw=pi/2 -> right (yaw=0 points up)

    rays = _scan(img, pose, num_rays=1, FOV=1, ray_length=100)
    assert rays.shape == (1, 2)

    x, y = map(int, rays[0])
    assert (x, y) == (img.shape[1] - 1, pose[1])


def test_obstacle_above_origin_collision_is_obstacle_cell() -> None:
    img = np.full((12, 12), 255, dtype=np.uint8)
    pose = (6, 9, 0.0)  # up (yaw=0 points up)
    img[4, 6] = 0  # obstacle directly above along same column

    rays = _scan(img, pose, num_rays=1, FOV=1, ray_length=200)
    x, y = map(int, rays[0])
    assert (x, y) == (6, 4)


def test_yaw_controls_direction_right_vs_left() -> None:
    img = np.full((11, 11), 255, dtype=np.uint8)
    pose = (5, 5)

    rays_right = _scan(img, (pose[0], pose[1], math.pi / 2),
                       num_rays=1, FOV=1, ray_length=200)
    xr, yr = rays_right[0]
    assert (xr, yr) == (img.shape[1] - 1, pose[1])

    rays_left = _scan(img, (pose[0], pose[1], -math.pi / 2),
                      num_rays=1, FOV=1, ray_length=200)
    xl, yl = rays_left[0]
    assert (xl, yl) == (0, pose[1])


def test_fov_180_default_yaw_first_left_last_right() -> None:
    img = np.full((21, 21), 255, dtype=np.uint8)
    pose = (10, 10)  # default yaw=0 (points up)

    rays = _scan(img, pose, num_rays=2, FOV=180, ray_length=500)
    (x0, y0) = map(int, rays[0])
    (x1, y1) = map(int, rays[-1])

    assert (x0, y0) == (0, pose[1])
    assert (x1, y1) == (img.shape[1] - 1, pose[1])


def test_fov_180_yaw_pi_over_2_first_top_last_bottom() -> None:
    img = np.full((21, 21), 255, dtype=np.uint8)
    pose = (10, 10, math.pi / 2)

    rays = _scan(img, pose, num_rays=2, FOV=180, ray_length=500)
    (x0, y0) = map(int, rays[0])
    (x1, y1) = map(int, rays[-1])

    assert (x0, y0) == (pose[0], 0)
    assert (x1, y1) == (pose[0], img.shape[0] - 1)


def test_pose_on_occupied_cell_raises() -> None:
    img = np.full((10, 10), 255, dtype=np.uint8)
    img[3, 4] = 0
    with pytest.raises(ValueError):
        _scan(img, (4, 3))


def test_rejects_non_integer_dtype() -> None:
    img = np.zeros((10, 10), dtype=np.float32)
    img[:] = 1.0
    with pytest.raises(ValueError):
        _scan(img, (5, 5))


def test_raises_when_image_dtype_is_not_np_integral() -> None:
    img = np.full((10, 10), 1.0, dtype=np.float64)
    with pytest.raises(ValueError):
        _scan(img, (5, 5))


def test_accepts_single_channel_3d_image() -> None:
    img = np.full((10, 10, 1), 255, dtype=np.uint8)
    rays = _scan(img, (5, 5, 0.0), num_rays=1, FOV=1, ray_length=50)
    assert rays.shape == (1, 2)


def test_only_true_collisions_true_returns_empty_when_no_collision() -> None:
    img = np.full((30, 30), 255, dtype=np.uint8)
    pose = (15, 15, 0.0)

    # Small ray_length keeps endpoints inside free space (no border hit, no obstacles)
    rays_true = _scan(
        img,
        pose,
        num_rays=5,
        FOV=30,
        ray_length=3,
        only_true_collisions=True,
    )
    assert rays_true.shape == (0, 2)

    rays_false = _scan(
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
    pose = (15, 15, math.pi / 2)

    # With num_rays=2 and FOV=180 around yaw=pi/2, angles are exactly -pi/2 (up) and +pi/2 (down)
    # Put an obstacle only on the "up" ray.
    img[13, 15] = 0  # (x=15, y=13)

    rays_true = _scan(
        img,
        pose,
        num_rays=2,
        FOV=180,
        ray_length=3,
        only_true_collisions=True,
    )
    assert rays_true.shape == (1, 2)
    assert tuple(map(int, rays_true[0])) == (15, 13)

    rays_false = _scan(
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


def test_uint8_conversion_warning_emitted_only_once() -> None:
    img = np.full((20, 20), 255, dtype=np.int32)
    pose = (10, 10, 0.0)

    # Make this test deterministic: clear the module warning registry so the first call warns.
    registry = getattr(rc_mod, "__warningregistry__", None)
    if isinstance(registry, dict):
        registry.clear()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default")

        _scan(img, pose, num_rays=1, FOV=1,
              ray_length=3, only_true_collisions=False)
        _scan(img, pose, num_rays=1, FOV=1,
              ray_length=3, only_true_collisions=False)

    conversion_warnings = [
        w
        for w in recorded
        if issubclass(w.category, UserWarning)
        and "Converting image of dtype" in str(w.message)
    ]
    assert len(conversion_warnings) == 1
