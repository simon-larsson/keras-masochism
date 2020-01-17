from setuptools import setup, find_packages
import masochism

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='keras-masochism',
    version=masochism.__version__,
    description='Masochistic deep learning with Keras.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    author='Simon Larsson',
    author_email='larssonsimon0@gmail.com',
    url='https://github.com/simon-larsson/keras-masochism',
    license='MIT',
    install_requires=[],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3 :: Only',
                 'Topic :: Scientific/Engineering']
)