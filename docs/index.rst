.. CodPy documentation master file, created by
   sphinx-quickstart on yyyy-mm-dd.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Codpy, the book !
===============================

This site is the companion site to the book *Reproducing Kernel Methods for Machine Learning, PDEs, and Statistics with Python*,
available on `arXiv <https://arxiv.org/abs/2402.07084>`_ by Philippe G. LeFloch, Jean-Marc Mercier, and Shohruh Miryusupov.

The main purpose of this companion site is to provide facilities to reproduce all figures of the book, 
in order to ease the diffusion of RKHS methods to teachers, students and practitioners. It allows to

- facilitate the understanding of the material presented in the book, and to provide a hands-on introduction to CodPy, an open-source Python library for numerical algorithms, machine learning, and computational statistics.
- enable readers to experiment with the algorithms presented in the book, and to modify the code for their own purposes.
- offer a platform for sharing additional resources, such as datasets, code snippets, and tutorials related to the book's content.


About CodPy
-----------
CodPy (Curse Of Dimensionality in Python) is an open-source, RKHS dedicated, Python library designed for numerical algorithms, machine learning, and computational statistics. It has a wide range of applications in finance, engineering, and industry, 
and its technical documentation is located at this `url <https://codpy.readthedocs.io/en/latest>`_.  

Installation
============

This has been tested on Windows 11.

1. Make a python 3.9 virtual environment

**Note:** Python 3.9 is required

.. code-block:: bash

   python --version
   python 3.9.13

Create a virtual environment:

.. code-block:: bash

   python -m venv venv
   venv\Scripts\activate

2. Clone the github repository:

.. code-block:: bash

   git clone https://github.com/smiryusupov2/codpybook.git

3. Install the dependencies

.. code-block:: bash

   cd codpybook
   pip install -e .

The python files can be found at: ``codpybook/docs/chX``

You can already run the files to reproduce the experiments available at `Readthedocs <https://codpybook-read-the-docs.readthedocs.io/en/latest/index.html>`_

4. Compiling with sphinx

If you want to re-compile all the experiments, as you see them on the `website <https://codpybook-read-the-docs.readthedocs.io/en/latest/index.html>`_, you can:

.. code-block:: bash

   cd docs
   sphinx-build html -b . _build

Once done, you can find the **index.html** file in: ``codpybook/docs/_build/index.html``

Alternatively, you can also install the package directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/smiryusupov2/codpybook.git

Report Issues
================
Please reposrt bug tracking and enhancement requests to our `GitHub page <https://github.com/JohnLeM/codpybook-rtd/issues>`_

.. toctree::
   :maxdepth: 2
   :caption: Chapter 2:
   :titlesonly:

   auto_ch2/index

.. toctree::
   :maxdepth: 2
   :caption: Chapter 3:
   :titlesonly:

   auto_ch3/index

.. toctree::
   :maxdepth: 2
   :caption: Chapter 4:
   :titlesonly:

   auto_ch4/index

.. toctree::
   :maxdepth: 2
   :caption: Chapter 5:
   :titlesonly:

   auto_ch5/index

.. toctree::
   :maxdepth: 2
   :caption: Chapter 6:
   :titlesonly:

   auto_ch6/index

.. toctree::
   :maxdepth: 2
   :caption: Chapter 7:
   :titlesonly:

   auto_ch7/index

.. toctree::
   :maxdepth: 2
   :caption: Chapter 8:
   :titlesonly:

   kqlearning
   auto_ch8/index

.. toctree::
   :maxdepth: 2
   :caption: Chapter 9:
   :titlesonly:

   auto_ch9/index 

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`





