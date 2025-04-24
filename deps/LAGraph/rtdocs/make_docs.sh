#!/bin/bash
READTHEDOCS=True sphinx-build . ./build
rm Doxyfile
python -c "import os, webbrowser; webbrowser.open('file://`pwd`/build/index.html')"