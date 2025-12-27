from setuptools import setup, Extension
import numpy as np

ext_modules = [
    Extension(
        "raycaster",
        sources=["src/raycast2D/_raycastermodule.c"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="raycaster",
    version="0.0.1",
    ext_modules=ext_modules,
)
