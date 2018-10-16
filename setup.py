#!/usr/bin/env python

from os import path
from setuptools import setup

description = ("Core libraries for natural language processing",)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(name="DetectorMorse",
      version="0.4.0",
      description="DetectorMorse, a sentence splitter",
      long_description=long_description,
      long_description_content_type="text/markdown",
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
