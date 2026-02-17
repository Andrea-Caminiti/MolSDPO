from setuptools import setup

setup(
    name="MolSDPO",
    version="0.0.1",
    packages=["MolSDPO"],
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.26.1",
        "ml-collections",
        "absl-py",
        "matplotlib",
        "torch>=2.8",
        "torch-geometric",
        "inflect==6.0.4",
        "pydantic==1.10.9",
        "hpsv2",
        "tqdm",
        "rdkit==2024.09.4",
        "openbabel-wheel",
        "lightning==2.0.3",
        "torchani",
    ]
    )