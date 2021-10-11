#!/bin/bash

echo "Removing __build folder" &&
rm -rf ./text/__build/ &&
echo "Building PDF" &&
jupyter-book build ./text/ --all --builder pdflatex &&
echo "Building HTML" &&
jupyter-book build ./text/ --all --builder html
