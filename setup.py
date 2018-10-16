#!/usr/bin/env python

from setuptools import setup

setup(name="DetectorMorse",
      version="0.4.0",
      description="DetectorMorse, a sentence splitter",
      author="Kyle Gorman",
      author_email="kylebgorman@gmail.com",
      packages=["detectormorse"],
      package_data={
          'detectormorse': ['models/*'],
      },
      install_requires=[
          'nlup>=0.7',
          'setuptools',  # For pkg_resources
      ],
      test_suite="default_model_test",
)
