#!/bin/bash

cd deps || exit
rm -r pybind
git clone https://github.com/pybind/pybind11
mv pybind11 pybind

rm -r eigen
git clone https://github.com/libigl/eigen

rm -r cereal
git clone https://github.com/USCiLab/cereal

cd .. || exit
pip install .
