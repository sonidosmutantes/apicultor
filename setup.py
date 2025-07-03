#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# For backward compatibility - main configuration is in pyproject.toml
setup(
    name='apicultor',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)