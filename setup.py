#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup, Extension
from setuptools import find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

examples = Extension('examples',
                    sources = ['playmag_freeze.sc'])

supercollider = Extension('apicultor.supercollider',
                    sources = ['live_coding.sc', 'setup_performance.sc'])

setup(name='apicultor',
      version='0.1.0',
      url='https://www.github.com/sonidosmutantes/apicultor',
      description='Another BigData approach to sound collections',
      long_description=long_description,
      author='Hernán Ordiales, Marcelo Tuller',
      author_email='hordiales@gmail.com, marscrophimself@protonmail.com',
      packages=find_packages(),
      license='GPLv3',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: everyone who likes music!',
        'Topic :: AI :: Music Information Retrieval :: Collaborative Composing and Remixing',

        'Programming Language :: Python :: 3.6',
    ],
    entry_points={'console_scripts': [
        'rpdl = apicultor.helper.WebScrapingDownload:main',
        'rpdla = apicultor.helper.ArchiveWebScrapingDownload:main',
        'miranalysis = apicultor.run_mir_analysis:main',
        'musicemotionmachine = apicultor.emotion.MusicEmotionMachine:main',
        'sonify = apicultor.sonification.Sonification:main',
        'qualify = apicultor.machine_learning.quality:main',
        'soundsimilarity = apicultor.machine_learning.SoundSimilarity:main',
        'mockrpapi = apicultor.MockRedPanalAPI_service:main',
        'audio2ogg = apicultor.helper.convert_to_ogg:main',
        'smcomposition = apicultor.state_machine.SMComposition:main'
    ]},
    install_requires=['numpy', 'wget', 'colorama', 'pysoundfile', 'librosa', 'scipy', 'matplotlib', 'scikit-learn', 'bs4', 'pandas']
     )
