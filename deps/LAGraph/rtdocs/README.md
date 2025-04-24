Automatic Deployment
======================
[LAGraph on ReadTheDocs](https://lagraph.readthedocs.io) is updated automatically when the
stable branch on github is updated. A webhook from GitHub to RTD activates, and RTD finds the
`.readthedocs.yaml` file to set up configuration and build and deploy the docs.

Local Build
===========
To preview changes to the documentation locally, first set up a python build environment
to run sphinx and doxygen.

## Option A: venv

**Note**: Only use this option if `cmake` and `doxygen` are already available on your system.
Otherwise, use Option B to `conda install` those libraries.

`venv` is an environment manager built in to Python 3. It will create an isolated environment
to `pip install` into without affecting the system python.

### Set up the virtual environment
*This only needs to be done once.*

    cd LAGraph/rtdocs
    python -m venv .venv
    source .venv/bin/activate
    pip install sphinx==4.0.3
    pip install sphinx_rtd_theme==0.5.2
    pip install breathe
    deactivate

### Build the docs

    cd LAGraph/rtdocs
    source .venv/bin/activate
    ./make_docs.sh


## Option B: conda
`conda` combines the functionality of both `venv` and `pip` and allows installing non-Python
packages in an environment.

If you don't already have conda installed, download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html).


### Set up the virtual environment

    conda create -n lagraph python=3.9
    conda activate lagraph
    conda install -y -c conda-forge doxygen cmake sphinx==4.0.3
    conda install -y -c conda-forge sphinx_rtd_theme==0.5.2 breathe
    conda deactivate

### Build the docs

    cd LAGraph/rtdocs
    conda activate lagraph
    ./make_docs.sh
