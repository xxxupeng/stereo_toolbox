#!/usr/bin/env python
#-*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    name = "stereo_toolbox",
    version = "0.1.0",
    keywords = ["pip", "stereo matching"],
    description = "stereo toolbox.",
    long_description = "A comprehensive stereo matching toolbox for efficient development and research.",
    license = "MIT Licence",

    url = "https://github.com/xxxupeng/stereo_toolbox",
    author = "xxxupeng",
    author_email = "xxxupeng@zju.edu.cn",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["numpy", "pillow", "torch", "opencv-python", "matplotlib", "jupyter", "matplotlib", "cmapy", "torchvision"]
)
