from . import raycaster
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import math


def cast(
    image: NDArray,
    pose: tuple[int, int] | tuple[int, int, float],
    num_rays: int = 1000,
    FOV: int = 360,
    ray_length: int = 500,
    only_true_collisions: bool = True,
) -> NDArray:
    """
    Cast 2D rays from a pose over a field-of-view and return ray endpoints.

    Args:
        image: Occupancy image of shape ``(H, W)`` or ``(H, W, 1)`` with an integer dtype.
            The raycaster treats ``0`` as occupied; the start cell at ``pose`` must be non-zero.
        pose: ``(x, y)`` or ``(x, y, yaw)`` where ``yaw`` is in radians.
        num_rays: Number of rays to cast (samples uniformly over the FOV).
        FOV: Field of view in degrees, centered around ``yaw``.
        ray_length: Max ray length in pixels.
        only_true_collisions: If True, return only rays that actually hit an obstacle
            (LiDAR-style). If False, return endpoints for all rays.

    Returns:
        An array of shape ``(N, 2)`` with dtype ``uint32`` containing ``(x, y)`` endpoints.
        If ``only_true_collisions=True``, ``N`` is the number of hits; otherwise ``N == num_rays``.
    """
    _check_args(image, pose)
    if len(image.shape) == 3:   # take only the first channel
        image = image[:, :, 0]

    p = _POSE(*pose)
    half_fov = math.radians(FOV) / 2.0
    start_angle = p.yaw - half_fov
    end_angle = p.yaw + half_fov

    theta = np.linspace(start_angle, end_angle, num_rays)
    x_rays = np.round(p.x + ray_length * np.cos(theta)).astype(np.uint32)
    y_rays = np.round(p.y + ray_length * np.sin(theta)).astype(np.uint32)
    
    rays = np.column_stack((x_rays, y_rays, np.zeros_like(y_rays)))
    
    # raycaster modifies the rays array inplace to include collisions
    raycaster.raycast(image.astype(np.uint8), rays, p.x, p.y)
    
    if only_true_collisions:    # LiDaR-style output
        return rays[rays[:, 2] == 1][:, :2]
    return rays[:, :2]


@dataclass
class _POSE:
    x: int
    y: int
    yaw: float = -math.pi / 2


def _check_args(img: NDArray, pose: tuple) -> None:
    if not np.issubdtype(img.dtype, np.integer):
        raise ValueError(
            f"Received an array with dtype {img.dtype}, The only supported dtypes are subtypes of np.integral.")

    shape = img.shape
    if len(shape) < 2 or (len(shape) > 2 and shape[2] > 1):
        raise ValueError(
            f"Received an array of shape {shape}. The only supported shapes are (H x W), (H x W x 1)."
        )
    if len(pose) < 2 or len(pose) > 3:
        raise ValueError(
            f"Received a pose tuple {pose}. The only supported pose formats are (x, y) and (x, y, yaw).")

    x, y = pose[0], pose[1]
    H, W = shape[0], shape[1]
    if x < 0 or x >= W:
        raise IndexError(
            f"x coordinate ({x}) if out of bounds for array with width {W}.")
    if y < 0 or y >= H:
        raise IndexError(
            f"y coordinate ({y}) if out of bounds for array with height {H}.")
        
    if img[y, x] == 0:
        raise ValueError(
            f"The pose {pose} corresponds to an occupied cell, cannot raycast from an occupied cell.")
