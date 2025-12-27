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
    ext_modules=ext_modules,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
