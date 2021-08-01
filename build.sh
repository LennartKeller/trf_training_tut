#!/bin/bash

echo "Removing __build folder" &&
rm -rf __build/ &&
echo "Building PDF" &&
jupyter-book build . --all --builder pdflatex &&
echo "Building HTML" &&
jupyter-book build . --all --builder html