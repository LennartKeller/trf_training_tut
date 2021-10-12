#!/bin/bash
echo "Running code formatting" &&
black . &&
echo "Removing __build folder" &&
jupyter-book clean ./text/ --all &&
echo "Building PDF" &&
jupyter-book build ./text/ --all --builder pdflatex &&
echo "Building HTML" &&
jupyter-book build ./text/ --all --builder html
