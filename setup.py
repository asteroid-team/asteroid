from setuptools import setup, find_packages

asteroid_version = "0.1.2"

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='asteroid',
    version=asteroid_version,
    author='Manuel Pariente',
    author_email='manuel.pariente@loria.fr',
    url="https://github.com/mpariente/asteroid",
    description='PyTorch-based audio source separation toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    python_requires='>=3.6',
    install_requires=['numpy',
                      'pyyaml',
                      'soundfile',
                      'scipy',
                      'torch',
                      'pytorch-lightning==0.6.0',
                      'pb_bss_eval',
                      'asranger',
                      'torch_stoi',
                      ],
    extras_require={
        'visualize': ['seaborn>=0.9.0'],
        'tests': ['pytest']
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)