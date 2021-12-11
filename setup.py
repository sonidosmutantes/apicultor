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

setup(name='apicultor-dev',
      version='2.0.1',
      url='https://www.github.com/sonidosmutantes/apicultor',
      description='BigData system of sound effects, remixes and sound collections',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Hernán Ordiales, Marcelo Tuller',
      author_email='hordiales@gmail.com, marscrophimself@protonmail.com',
      download_url='https://github.com/sonidosmutantes/apicultor/archive/refs/heads/dev.zip',
      keywords = ['deeplearning', 'ml', 'mlops', 'svm', 'music', 'dsp', 'sound', 'audio'],      
      packages=find_packages(),
      license='GPLv3',
    classifiers=[
        'Development Status :: 6 - Mature',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3.9',
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
    install_requires=['numpy', 'numba', 'smst', 'wget', 'colorama', 'transitions','pysoundfile', 'librosa', 'scipy', 'matplotlib', 'scikit-learn', 'bs4', 'pandas']
     )
