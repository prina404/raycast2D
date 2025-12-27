from setuptools import setup, Extension, find_packages
import numpy as np

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:
    import tomli as tomllib  # py<=3.10


def read_project_metadata():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    proj = data.get("project", {})
    return proj.get("name"), proj.get("version")


name, version = read_project_metadata()

ext_modules = [
    Extension(
        "raycast2D.raycaster",
        sources=["src/raycast2D/_raycastermodule.c"],
        include_dirs=[np.get_include()],
    )
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=name,
    version=version,
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
