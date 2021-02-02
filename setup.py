from setuptools import setup, find_namespace_packages

# defines the projects package namespace for processing
setup(
    name='m2lib',
    version='0.1',
    packages=find_namespace_packages(include=['m2lib.*'])
)

# should we add additional packages????