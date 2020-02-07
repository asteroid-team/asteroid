from setuptools import setup

asteroid_version = "0.0.1"

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='asteroid',
    version=asteroid_version,
    author='Manuel Pariente',
    author_email='manuel.pariente@loria.fr',
    url="https://github.com/mpariente/AsSteroid",
    description='PyTorch-based audio source separation toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    python_requires='>=3.6',
    install_requires=['numpy',
                      'Cython',
                      'scipy',
                      'pandas',
                      'pyyaml',
                      'soundfile',
                      'torch',
                      'torchvision',
                      'pytorch-lightning'],
    extras_require={
        'visualize': ['seaborn>=0.9.0'],
        'tests': ['pytest']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)