[![downloads](https://img.shields.io/pypi/dm/pyratlib?color=blue&style=flat-square)](https://pypi.org/project/pyratlib/)
[![version](https://img.shields.io/pypi/v/pyratlib?color=blue&style=flat-square)](https://pypi.org/project/pyratlib/)
[![commit](https://img.shields.io/github/last-commit/pyratlib/pyrat?color=blue&style=flat-square)](https://github.com/pyratlib/pyrat/commits/main)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/pyrat/pyratlib)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5883277.svg)](https://doi.org/10.5281/zenodo.5883277)
[![Stars](https://img.shields.io/github/stars/pyratlib/pyrat?style=social)](https://github.com/pyratlib/pyrat/stargazers)

<!-- [![Build Status](https://img.shields.io/appveyor/build/pyrat/pyratlib?style=flat-square)](https://travis-ci.com/pyrat/pyratlib) -->

<p align="center">
  <img width="500" height="600" src="https://github.com/pyratlib/pyrat/blob/main/docs/LOGO%20PYRAT.png">
</p>


# PyRat - Python in Rodent Analysis and Tracking
------------
PyRat is a user friendly library in python to analyze data from the DeepLabCut. Developed to help researchers unfamiliar with programming can perform animal behavior analysis more simpler.

# Installation
------------

The latest stable release is available on PyPi, and you can install it by saying
```
pip install pyratlib
```
Anaconda users can install using ``conda-forge``:
```
conda install -c conda-forge pyratlib
```

To build PyRat from source, say `python setup.py build`.
Then, to install PyRat, say `python setup.py install`.
If all went well, you should be able to execute the demo scripts under `examples/`
(OS X users should follow the installation guide given below).

Alternatively, you can download or clone the repository and use `pip` to handle dependencies:

```
unzip pyratlib.zip
pip install -e pyratlib
```
or
```
git clone https://github.com/pyratlib/pyrat.git
pip install -e pyratlib
```

By calling `pip list` you should see `pyrat` now as an installed package:
```
pyrat (0.x.x, /path/to/pyratlib)
```
# Data
------

The data is available on [Zenodo](https://doi.org/10.5281/zenodo.5865893)

# Examples
-----------
<!-- 
- Notebook with the t-SNE algorithm. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tuliofalmeida/pyjama/blob/main/PyJama_JAMA_exemple.ipynb)       -->
- Basic Usage [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyratlib/pyrat/blob/main/PyRAT_Basic_Plots.ipynb)
- Behavior Classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyratlib/pyrat/blob/main/PyRAT_Behavior_Classification.ipynb)
- Metrics in mice [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyratlib/pyrat/blob/main/PyRAT_Mice.ipynb)
- Neural Data example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/pyratlib/pyrat/blob/main/PyRAT_Neural_Data.ipynb)

# References:
-----------

If you use our code or data we kindly as that you please cite [De Almeida et al, 2022](https://www.frontiersin.org/articles/10.3389/fnins.2022.779106/full) and, if you use the Python package (DeepLabCut2.x) please also cite [De Almeida et al, 2021](https://zenodo.org/record/5865893).

- De Almeida et al, 2022: [https://doi.org/10.3389/fnins.2022.779106](https://doi.org/10.3389/fnins.2022.779106)
- De Almeida et al, 2021: [10.5281/zenodo.5883277](https://zenodo.org/record/5883277)


Please check out the following references for more details:

    @article{deAlmeida2022,
      title={PyRAT: An open source-python library for fast and robust animal behavior analysis and neural data synchronization},
      author={De Almeida, Tulio Fernandes and Spinelli, Bruno Guedes and Hypolito Lima, Ram{\'o}n and Gonzalez, Maria Carolina and Rodrigues, Abner Cardoso},
      journal={Frontiers in Neuroscience},
      pages={505},
      publisher={Frontiers}
    }

    @dataset{deAlmeida2021,
      author       = {Almeida, Túlio and
                      Spinelli, Bruno and
                      Gonzalez, Maria Carolina and
                      Lima, Ramón and
                      Rodrigues, Abner},
      title        = {PyRAT-data-example},
      month        = sep,
      year         = 2021,
      publisher    = {Zenodo},
      version      = {1.0.0},
      doi          = {10.5281/zenodo.5883277},
      url          = {https://doi.org/10.5281/zenodo.5883277}
    }

# Development Team:
------------

- Tulio Almeida - [GitHub](https://github.com/tuliofalmeida) - [Google Scholar](https://scholar.google.com/citations?user=kkOy-JkAAAAJ&hl=en)
- Bruno Spinelli - [GitHub](https://github.com/brunospinelli) - [Google Scholar](https://scholar.google.com/)
- Ramon Hypolito - [GitHub](https://github.com/ramonhypolito) - [Google Scholar](https://scholar.google.com/citations?user=5lKx5GcAAAAJ&hl=pt-BR&oi=ao)
- Maria Carolina Gonzalez - [GitHub](https://github.com/pyratlib) - [Google Scholar](https://scholar.google.com/citations?user=7OXkSPcAAAAJ&hl=pt-BR&oi=ao)
- Abner Rodrigues - [GitHub](https://github.com/abnr) - [Google Scholar](https://scholar.google.com.br/citations?user=0dTid9EAAAAJ&hl=en)


