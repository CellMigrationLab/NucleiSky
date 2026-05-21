from setuptools import setup, find_packages

setup(
    name="nucleisky",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12.13",
    install_requires=[
    ],
)
