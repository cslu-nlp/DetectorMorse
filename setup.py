#!/usr/bin/env python

from setuptools import setup

setup(name="DetectorMorse",
      version="0.3.0",
      description="DetectorMorse, a sentence splitter",
      author="Kyle Gorman",
      author_email="kylebgorman@gmail.com",
      packages=["detectormorse"],
      package_data = {
          'detectormorse': ['models/*'],
      },
      install_requires=[
          'nlup>=0.5.0',
          'setuptools',  # For pkg_resources
      ],
)
