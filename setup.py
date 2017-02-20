#! /usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

setup(name='apicultor',
      description='Extrae miel de RedPanal',
      author='Hernán Hordiales, Marcelo Tuller',
      author_email='hordiales@gmail.com, marscrophimself@protonmail.com',
      packages=['apicultor', 'apicultor.data', 'apicultor.tests', 'apicultor.utils', 'apicultor.constraints', 'apicultor.gradients'],
     )
