from pathlib import Path
import numpy as np
import pygame
from PIL import Image
import os
import time

from raycast2D import cast


NUM_RAYS = 300
FOV = 360
RAY_LENGTH = 200
FRAME_AVG_WINDOW = 60


def load_binary_map(path, threshold=100):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    arr = np.where(arr < threshold, 0, 255).astype(np.uint8)
    return arr


def main():
    fp = Path(__file__).parents[1]
    grid = load_binary_map(fp / "media/lab_intel.png", threshold=100)
    h, w = grid.shape 

    pygame.init()
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    # Pre-render the map as a surface (so we don't blit pixel-by-pixel each frame)
    # White = free, black = obstacle (adjust if your convention differs)
    map_rgb = np.stack([grid, grid, grid], axis=-1)  # (H,W,3)
    map_surface = pygame.surfarray.make_surface(
        map_rgb.swapaxes(0, 1))  # pygame expects (W,H,3)

    ema_ms = None
    ema_alpha = 2.0 / (FRAME_AVG_WINDOW + 1.0)

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                pygame.quit()
                raise SystemExit

        mx, my = pygame.mouse.get_pos()

        try:    # avoid exceptions when mouse is on obstacle
            t0 = time.perf_counter()
            rays = cast(grid, (mx, my), num_rays=NUM_RAYS,
                        FOV=FOV, ray_length=RAY_LENGTH, only_true_collisions=False)
            dt_ms = (time.perf_counter() - t0) * 1000.0
        except: 
            continue
        ema_ms = dt_ms if ema_ms is None else (
            ema_ms + ema_alpha * (dt_ms - ema_ms))

        screen.blit(map_surface, (0, 0))

        # Draw origin
        pygame.draw.circle(screen, (0, 255, 0), (mx, my), 3)

        for r in rays:
            x1, y1 = r
            pygame.draw.line(screen, (255, 0, 0), (mx, my), (x1, y1), 1)
            pygame.draw.circle(screen, (0, 0, 255),
                               (int(x1), int(y1)), 2)

        pygame.display.set_caption(
            f"raycast2D demo | raycast frametime {ema_ms:.3f} ms (window={FRAME_AVG_WINDOW})"
        )

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
