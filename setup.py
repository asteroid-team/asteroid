from setuptools import setup

setup(
    name='asteroid',
    version='0.0.1',
    description='Source separation on steroids',
    author='Manuel Pariente',
    author_email='manuel.pariente@loria.fr',
    license='MIT',
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'pyyaml',
                      'soundfile',
                      'torch',
                      'torchvision',
                      'pytorch-lightning'],
    extra_requires={
        'visualize': ['seaborn>=0.9.0'],
        'evaluate': ['mir-eval>=0.5', 'pystoi>=0.2.2', 'pypesq>=1.0'],
        'tests': ['pytest']
    }
)