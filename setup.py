from setuptools import setup

setup(
    name="MolSDPO",
    version="0.0.1",
    packages=["MolSDPO"],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "ml-collections",
        "absl-py",
        "matplotlib",
        "torch>=2.8",
        "torch-geometric",
        "inflect",
        "pydantic",
        "hpsv2",
        "tqdm",
        "rdkit",
        "openbabel-wheel",
        "lightning",
        "diffusers",
        "posebusters"
    ]
    )
