from typing import Optional
from . import raycaster
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import math
import warnings


class Lidar2D:

    def __init__(
        self,
        num_rays: int = 1000,
        FOV: int = 360,
        ray_length: int = 500,
        occupancy_grid: NDArray | None = None,
        obstacle_threshold: int = 0,
    ) -> None:
        """
        Initialize the Lidar2D raycaster.
        Args:
            num_rays: Number of rays to cast (samples uniformly over the FOV).
            FOV: Field of view in degrees.
            ray_length: Max ray length in pixels.
            occupancy_grid: Optional occupancy grid to set at initialization.
            obstacle_threshold: Cells with values less than or equal to this are treated as occupied.
        """
        self.angles = np.linspace(
            -math.radians(FOV) / 2.0,
            math.radians(FOV) / 2.0,
            num_rays,
        )
        self.ray_length = ray_length
        self.obstacle_threshold = obstacle_threshold
        self._map = None
        if occupancy_grid is not None:
            self.set_map(occupancy_grid)

    def scan(
        self,
        pose: tuple[int, int] | tuple[int, int, float],
        image: Optional[NDArray] = None,
        only_true_collisions: bool = True
    ) -> NDArray[np.uint32]:
        """
        Perform a LiDAR scan from the given pose on the occupancy image.

        Args:
            pose: ``(x, y)`` or ``(x, y, yaw)`` where ``yaw`` is in radians.
            image: Optional occupancy grid of shape ``(H, W)`` or ``(H, W, 1)`` with an integer dtype.
                The raycaster treats values less than or equal to ``obstacle_threshold`` as occupied;
                the start cell at ``pose`` must be free.
                NOTE: the provided image overrides any occupancy grid set at initialization.
            only_true_collisions: If True, return only rays that actually hit an obstacle
                (LiDAR-style). If False, return endpoints for all rays.

        Returns:
            An array of shape ``(N, 2)`` with dtype ``uint32`` containing ``(x, y)`` collision coordinates.
        """
        if self._map is None and image is None:
            raise ValueError(
                "No occupancy grid provided at initialization or scan time.")
            
        if image is not None :
            _map = (_check_map(image) > self.obstacle_threshold).astype(np.uint8)
        else:
            _map = self._map
            
        _check_pose(_map, pose)

        p = _POSE(*pose)  # type: ignore
        x_rays = np.round(p.x + self.ray_length *
                          np.cos(self.angles + p.yaw)).astype(np.uint32)
        y_rays = np.round(p.y + self.ray_length *
                          np.sin(self.angles + p.yaw)).astype(np.uint32)

        rays = np.column_stack((x_rays, y_rays, np.zeros_like(y_rays)))

        # raycaster modifies the rays array inplace to include collisions
        raycaster.raycast(_map, rays, p.x, p.y)

        if only_true_collisions:    # LiDaR-style output
            return rays[rays[:, -1] == 1][:, :-1]
        return rays[:, :-1]

    def set_map(self, occupancy_grid: NDArray) -> None:
        """
        Set or update the occupancy grid used by the raycaster.

        Args:
            occupancy_grid: Occupancy grid of shape ``(H, W)`` or ``(H, W, 1)`` with an integer dtype.
        """
        self._map = _check_map(occupancy_grid)
        self._map = (self._map > self.obstacle_threshold).astype(np.uint8)


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
        An array of shape ``(N, 2)`` with dtype ``uint32`` containing ``(x, y)`` collision coordinates.
        If ``only_true_collisions=True``, ``N`` is the number of hits; otherwise ``N == num_rays``.
    """
    image = _check_map(image)
    _check_pose(image, pose)

    p = _POSE(*pose)  # type: ignore
    half_fov = math.radians(FOV) / 2.0
    start_angle = p.yaw - half_fov
    end_angle = p.yaw + half_fov

    theta = np.linspace(start_angle, end_angle, num_rays)
    x_rays = np.round(p.x + ray_length * np.cos(theta)).astype(np.uint32)
    y_rays = np.round(p.y + ray_length * np.sin(theta)).astype(np.uint32)

    rays = np.column_stack((x_rays, y_rays, np.zeros_like(y_rays)))

    # raycaster modifies the rays array inplace to include collisions
    raycaster.raycast(image, rays, p.x, p.y)

    if only_true_collisions:    # LiDaR-style output
        return rays[rays[:, -1] == 1][:, :-1]
    return rays[:, :-1]


@dataclass
class _POSE:
    x: int
    y: int
    yaw: float = 0.0
    
    def __post_init__(self):
        self.yaw -= math.pi / 2.0  # adjust so yaw=0 points upwards

def _check_pose(img: NDArray[np.uint8], pose: tuple) -> None:
    if len(pose) < 2 or len(pose) > 3:
        raise ValueError(
            f"Received a pose tuple {pose}. The only supported pose formats are (x, y) and (x, y, yaw).")

    x, y = pose[0], pose[1]
    H, W = img.shape[0], img.shape[1]
    if x < 0 or x >= W:
        raise IndexError(
            f"x coordinate ({x}) if out of bounds for array with width {W}.")
    if y < 0 or y >= H:
        raise IndexError(
            f"y coordinate ({y}) if out of bounds for array with height {H}.")

    if img[y, x] == 0:
        raise ValueError(
            f"The pose {pose} corresponds to an occupied cell, cannot raycast from an occupied cell.")


def _check_map(img: NDArray) -> NDArray[np.uint8]:
    if not np.issubdtype(img.dtype, np.integer):
        raise ValueError(
            f"Received an array with dtype {img.dtype}, The only supported dtypes are subtypes of np.integral.")

    shape = img.shape
    if len(shape) < 2 or (len(shape) > 2 and shape[2] > 1):
        raise ValueError(
            f"Received an array of shape {shape}. The only supported shapes are (H x W), (H x W x 1)."
        )

    if len(img.shape) == 3:   # take only the first channel
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        warnings.warn(
            f"Converting image of dtype {img.dtype} to uint8 for raycasting.")
        img = img.astype(np.uint8)

    return img
