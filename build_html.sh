#!/bin/bash
printf "\n\nRunning code formatting\n\n" &&
black . &&
printf "\n\nRemoving old html build\n\n" &&
jupyter-book clean ./text/ --html &&
printf "\n\nBuilding HTML\n\n" &&
jupyter-book build ./text/ --all --builder html
