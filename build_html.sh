#!/bin/bash
echo "Running code formatting" &&
black . &&
echo "Removing old html build" &&
jupyter-book clean ./text/ --html &&
echo "Building HTML" &&
jupyter-book build ./text/ --all --builder html
