#!/bin/bash
printf "\n\nRunning code formatting\n\n" &&
black . &&
printf "\n\nRemoving __build folder\n\n" &&
jupyter-book clean ./text/ --all &&
printf "\n\nBuilding PDF\n\n" &&
jupyter-book build ./text/ --all --builder pdflatex &&
printf "\n\nBuilding HTML\n\n" &&
jupyter-book build ./text/ --all --builder html
