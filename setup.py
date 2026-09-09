from setuptools import setup, find_packages

setup(
    name="t2cbench",
    version="0.1.0",
    description="A fair benchmark for text-to-CAD generation",
    packages=find_packages(include=["t2cbench", "t2cbench.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24", "scipy>=1.10", "pandas>=2.0", "trimesh>=4.0",
        "matplotlib>=3.7", "tqdm>=4.65", "pyyaml>=6.0", "tabulate>=0.9",
        "huggingface_hub>=0.23", "rtree>=1.0", "embreex",
    ],
)
