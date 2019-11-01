from setuptools import setup

setup(
    name='asteroid',
    version='0.0.1',
    description='Source separation on steroids',
    author='Manuel Pariente',
    author_email='manuel.pariente@loria.fr',
    license='MIT',
    install_requires=['numpy',
                      'torch',
                      'soundfile']
)
