#!/bin/bash
# Install conda for python 3.8 on a 64 bit system
# https://docs.conda.io/en/latest/miniconda.html?ref=learn-ubuntu#linux-installers

# Download the installer
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh

# check the hash
hash="640b7dceee6fad10cb7e7b54667b2945c4d6f57625d062b2b0952b7f3a908ab7"
echo "$hash  Miniconda3-py38_23.1.0-1-Linux-x86_64.sh" | sha256sum -c

# Run the installer
#bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
