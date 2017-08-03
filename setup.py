#!/usr/bin/env python
from setuptools import setup, find_packages
from codecs import open
from os import path


with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup_info = dict(
     name='cove',
     version='1.0.0',
     author='Bryan McCann',
     author_email='Bryan.McCann.is@gmail.com',
     url='https://github.com/salesforce/cove',
     description='Context Vectors for Deep Learning and NLP',
     long_description=long_description,
     license='BSD 3-Clause',
     keywords='cove, context vectors, deep learning, natural language processing',
     packages=find_packages()
)

setup(**setup_info)
