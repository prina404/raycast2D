"""Simple benchmarks for raycast2D.

Runs a small, fixed benchmark matrix (≈50 cases) and prints throughput in rays/s.

Usage:
    python3 test/benchmark.py
"""
import math
import time
import numpy as np
from raycast2D import cast
from itertools import product

# Fixed benchmark configuration
CONFIG = {
    "sizes": [512, 4096, 8192],
    "num_rays": [500, 1500, 5000],
    "ray_length": [500, 2000],
    "densities": [0.4,],
    "fov": 360,
    "yaw": 0.0,
    "repeats": 3,
    "seed": 12345,
}


def make_map(size: int, density: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Start all free.
    grid = np.full((size, size), 255, dtype=np.uint8)

    scale = 2**32
    threshold = int(density * scale)
    for y in range(size):
        row = rng.integers(0, scale, size=size, dtype=np.uint32)
        grid[y, row < threshold] = 0
    return grid


def choose_pose_free_center(grid: np.ndarray, *, yaw: float) -> tuple[int, int, float]:
    """Pick a deterministic pose near center and force it to be free."""
    h, w = grid.shape[:2]
    x = w // 2
    y = h // 2
    grid[y, x] = 255  # free
    return (x, y, float(yaw))


def _time_cast(
        grid: np.ndarray,
        pose: tuple[int, int, float],
        *,
        num_rays: int,
        fov: int,
        ray_length: int,
        repeats: int,
) -> tuple[list[float], np.ndarray]:
    """Return per-call durations in ms, plus the last rays array."""
    last_rays: np.ndarray | None = None

    timings_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        last_rays = cast(grid, pose, num_rays=num_rays,
                         FOV=fov, ray_length=ray_length)
        t1 = time.perf_counter_ns()
        timings_ms.append((t1 - t0) / 1_000_000.0)

    assert last_rays is not None
    return timings_ms, last_rays


def main() -> None:
    sizes = CONFIG["sizes"]
    densities = CONFIG["densities"]
    rays_list = CONFIG["num_rays"]
    ray_lengths = CONFIG["ray_length"]
    fov = int(CONFIG["fov"])
    yaw = float(CONFIG["yaw"])
    repeats = int(CONFIG["repeats"])
    seed0 = int(CONFIG["seed"])

    cases = []
    for size, density, ray_length, num_rays in product(
        sizes, densities, ray_lengths, rays_list
    ):
        cases.append(
            {
                "size": int(size),
                "density": float(density),
                "ray_length": int(ray_length),
                "num_rays": int(num_rays),
                "fov": fov,
                "yaw": yaw,
            }
        )

    print(f"cases={len(cases)} repeats={repeats}")

    header = (
        f"{'size':>6} "
        f"{'dens':>6} "
        f"{'ray_len':>6} "
        f"{'rays':>7} "
        f"{'total_ms':>9} "
        f"{'rays/s':>12}"
    )
    print(header)
    print("-" * len(header))

    for idx, c in enumerate(cases, start=1):
        grid = make_map(c["size"], c["density"], seed=seed0 + idx)
        pose = choose_pose_free_center(grid, yaw=c["yaw"])

        timings_ms, rays = _time_cast(
            grid,
            pose,
            num_rays=c["num_rays"],
            fov=c["fov"],
            ray_length=c["ray_length"],
            repeats=repeats,
        )

        if rays.shape != (c["num_rays"], 2):
            raise RuntimeError(f"Unexpected rays shape: {rays.shape}")

        mean_ms = sum(timings_ms) / len(timings_ms)
        rays_per_s = (c["num_rays"] * 1000.0 /
                      mean_ms) if mean_ms > 0 else math.inf

        print(
            f"{c['size']:>6d} "
            f"{c['density']:>6.3f} "
            f"{c['ray_length']:>6d} "
            f"{c['num_rays']:>7d} "
            f"{mean_ms:>9.4f} "
            f"{rays_per_s:>12.2f}"
        )


if __name__ == "__main__":
    main()
