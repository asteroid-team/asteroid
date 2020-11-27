from setuptools import setup, find_packages

asteroid_version = "0.4.0rc1"

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="asteroid",
    version=asteroid_version,
    author="Manuel Pariente",
    author_email="manuel.pariente@loria.fr",
    url="https://github.com/mpariente/asteroid",
    description="PyTorch-based audio source separation toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        "pyyaml",
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "pytorch-lightning>=0.8,<0.10.0",
        "torch_optimizer",
        "soundfile",
        "pb_bss_eval",
        "torch_stoi",
        "asteroid-filterbanks",
    ],
    extras_require={
        "tests": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "asteroid-upload=asteroid.scripts.asteroid_cli:upload",
            "asteroid-infer=asteroid.scripts.asteroid_cli:infer",
            "asteroid-register-sr=asteroid.scripts.asteroid_cli:register_sample_rate",
            "asteroid-versions=asteroid.scripts.asteroid_versions:print_versions",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
