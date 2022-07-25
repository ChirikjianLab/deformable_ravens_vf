#!/bin/bash
set -uexo pipefail

echo "Installing Python libraries..."

pip install pybullet==3.2.0
pip install packaging==21.3
pip install matplotlib==3.1.2
pip install opencv-python==4.5.5.62
pip install meshcat==0.3.2
pip install transformations==2021.6.6
pip install scikit-image==0.16.2
pip install tensorflow-addons==0.15.0
pip install tensorflow==2.8.0

pip install -e .
