# raycast2D

[![PyPI](https://img.shields.io/pypi/v/raycast2D.svg)](https://pypi.org/project/raycast2D/)
[![Python](https://img.shields.io/pypi/pyversions/raycast2D.svg)](https://pypi.org/project/raycast2D/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-informational)](https://pypi.org/project/raycast2D/)
[![Build](https://github.com/prina404/raycast2D/actions/workflows/build.yml/badge.svg)](https://github.com/prina404/raycast2D/actions/workflows/build.yml)
[![Tests](https://github.com/prina404/raycast2D/actions/workflows/tests.yml/badge.svg)](https://github.com/prina404/raycast2D/actions/workflows/tests.yml)

`raycast2D` is a fast, single-core 2D raycasting implementation written in C with a small Python API.
It operates on NumPy occupancy grids / binary images (free cells are non-zero; occupied cells are `0`) and computes ray intersections using Bresenham line algorithm.

The core package depends only on `numpy`. Optional extras are provided for the interactive demo and development.

![](media/raycast_compressed.gif)

## Installation & Usage

```bash
$ pip install raycast2D
```
### Basic raycast on an image

```python
import numpy as np
from raycast2D import cast
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("<path/to/img.png>")
img_array = np.array(img)
img_array = img_array[:, :, 0].astype(np.uint8)  # Use single channel
# img_array = img_array[img_array < 100] = 0  # threshold obstacles if needed

rays = cast(img_array, pose=(250, 250), num_rays=360, ray_length=500)
# rays is an array of shape (N, 2) where each row contains the (x, y) coordinates of the ray collisions

plt.imshow(img_array, cmap='gray')
plt.scatter([250], [250], c='green', s=10)
plt.scatter(rays[:, 0], rays[:, 1], c='blue', s=1)
for ray in rays:
    plt.plot([250, ray[0]], [250, ray[1]], c='red', linewidth=0.5, alpha=0.3)
plt.show()
```

## Performance

The benchmark script in `test/benchmark.py` runs a small set of examples and prints throughput in rays/s.
Example results (tested on an i7-9700k):

|  Map size | Ray length | Rays per call | Mean time (ms) | Throughput (rays/s) |
| --------: | ---------: | ------------: | -------------: | ------------------: |
|   512×512 |       2000px |          5000 |         0.2904 |          17,216,880 |
| 4096×4096 |       2000px |          5000 |         1.7626 |           2,836,661 |
| 8192×8192 |       2000px |          5000 |         7.3452 |             680,719 |

To reproduce on your machine:

```bash
$ python3 test/benchmark.py
```



## Interactive demo

The repository includes an interactive `pygame` demo that raycasts from the current mouse position.

```bash
$ pip install "raycast2D[extra]"
$ python3 test/demo.py
```

## Testing and development

Install the test/development dependencies:

```bash
$ pip install "raycast2D[test]"
```

Run the test suite with:

```bash
pytest
```

## License

GPL-3.0-only. See [LICENSE](LICENSE).
