# raycast2D

[![PyPI](https://img.shields.io/pypi/v/raycast2D.svg)](https://pypi.org/project/raycast2D/)
[![Python](https://img.shields.io/pypi/pyversions/raycast2D.svg)](https://pypi.org/project/raycast2D/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-informational)](https://pypi.org/project/raycast2D/)
[![Build](https://github.com/prina404/raycast2D/actions/workflows/build.yml/badge.svg)](https://github.com/prina404/raycast2D/actions/workflows/build.yml)
[![Tests](https://github.com/prina404/raycast2D/actions/workflows/tests.yml/badge.svg)](https://github.com/prina404/raycast2D/actions/workflows/tests.yml)

`raycast2D` is a fast, single-thread 2D raycasting implementation written in C with a small Python API.
It operates on NumPy occupancy grids / binary images (free cells are non-zero; occupied cells are `0`) and computes ray intersections using Bresenham line algorithm.

The core package depends only on `numpy`.

![](media/raycast_compressed.gif)

## Installation & Usage

```bash
$ pip install raycast2D
```
### Basic `cast` function usage

```python
import numpy as np
from PIL import Image
from raycast2D import cast

img = Image.open("<path/to/img.png>")
img_array = np.array(img)[:, :, 0] # Use single channel
pose = (250, 250, 0.5)  # x, y, yaw in radians 

rays = cast(img_array, pose, num_rays=360, ray_length=500)

print(rays[:5])
# Expected output (example):
# [[x0 y0]
#  [x1 y1]
#  [x2 y2]
#  [x3 y3]
#  [x4 y4]]
```

### Using the `Lidar2D` class

```python
from raycast2D import Lidar2D
# [...] 
lidar = Lidar2D(num_rays=360, FOV=360, ray_length=500)
lidar.set_map(img_array)

rays = lidar.scan((250, 250, 0.5))

print(rays[:5])
# same output as before...
```

### Plotting example

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from raycast2D import cast

img = Image.open("media/lab_intel.png")
img_array = np.array(img)[:, :, 0] # Use single channel

pose = (350, 350, -1.0)
rays = cast(img_array, pose=pose, num_rays=200, FOV=90, ray_length=250)

plt.imshow(img_array, cmap="gray")
plt.scatter([pose[0]], [pose[1]], c="green", s=10)
plt.scatter(rays[:, 0], rays[:, 1], c="blue", s=1)
for ray in rays:
    plt.plot([pose[0], ray[0]], [pose[1], ray[1]], c="red", linewidth=0.5, alpha=0.3)
plt.show()
```

![](media/example.png)

## Performance

The benchmark script in `test/benchmark.py` runs a small set of examples and prints throughput in rays/s.
Example results (tested on an i7-9700k):

|  Map size | Ray length | Rays per call | Mean time (ms) | Throughput (rays/s) |
| --------: | ---------: | ------------: | -------------: | ------------------: |
|   512×512 |     2000px |          5000 |         0.2490 |          20,083,924 |
| 4096×4096 |     2000px |          5000 |         0.2848 |          17,556,549 |
| 8192×8192 |     2000px |          5000 |         0.2571 |          19,446,324 |

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
