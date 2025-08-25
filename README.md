# Reproducing kernel methods for machine learning, PDEs, and statistics with CodPy

This repository accompanies the book  
**[Reproducing kernel methods for machine learning, PDEs, and statistics with Python](https://arxiv.org/)**  
by *Philippe G. LeFloch, Jean-Marc Mercier, and Shohruh Miryusupov* (available on arXiv).  

It provides **Python code implementations** and **reproductions of the experiments** detailed in the book using the [CodPy library](https://github.com/smiryusupov2/codpy/tree/main).

---

## About CodPy

**CodPy (Curse Of Dimensionality in Python)** is an open-source Python library for:

- Numerical algorithms  
- Machine learning  
- Computational statistics  

It has applications in **finance, engineering, and industry**.  

This repository focuses on **reproducing the experiments from the book** chapter by chapter.

---

## Installation
This has been tested on Windows 11. 

1. Make a python 3.9 virtual environment

Note: Python 3.9 is required
```bash 
python --version
python 3.9.13
```

Create a virtual environment: 
```bash 
python -m venv venv
venv\Scripts\activate
```

2. Clone the github repository: 
```bash 
git clone https://github.com/JohnLeM/codpybook-rtd.git
```

3. Install the dependencies
```bash 
cd codpybook-rtd
pip install -e .
```

The python files can be found at: ```codpybook-rtd/docs/chX```
You can already run the files to reproduce the experiments available at [Readthedocs](https://codpybook-read-the-docs.readthedocs.io/en/latest/index.html)

4. Compiling with sphinx
If you want to re-compile all the experiments, as you see them on the [website](https://codpybook-read-the-docs.readthedocs.io/en/latest/index.html), you can: 
```bash 
cd docs
sphinx-build html -b . _build 
```
Once done, you can find the **index.html** file in: ```codpybook-rtd/docs/_build/index.html```

Alternatively, you can also install the package directly from GitHub:

```bash
pip install git+https://github.com/JohnLeM/codpybook-rtd.git
```

## Report Issues
If you encounter any issues or have suggestions for improvements, please report them on the [GitHub Issues page](https://github.com/JohnLeM/codpybook-rtd/issues)


# License
© 2024 Philippe G. LeFloch, Jean-Marc Mercier, and Shohruh Miryusupov.  
This project is licensed under the [MIT License](LICENSE).