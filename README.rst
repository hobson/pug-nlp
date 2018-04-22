=======
pug-nlp
=======

|Build Status| |Coverage Status| |Version Status| |Downloads|

PUG Natural Language Processing (NLP) Utilities
-----------------------------------------------

This sub-package of the pug namespace package, provides natural language
processing (NLP) and text processing utilities built by and for the PDX
Python User Group (PUG).

Description
===========

Installation
------------

On a Posix System
~~~~~~~~~~~~~~~~~

You really want to contribute, right?

::

    git clone https://github.com/totalgood/pug-nlp.git

If you're a user and not a developer, and have an up-to-date posix OS
with the postgres, xml2, and xlst development packages installed, then
just use ``pip``.

::

    pip install pug-nlp

Fedora
~~~~~~

If you're on Fedora >= 16 but haven't done a lot of python binding
development, then you'll need some libraries before pip will succeed.

::

    sudo yum install -y python-devel libxml2-devel libxslt-devel gcc-gfortran python-scikit-learn postgresql postgresql-server postgresql-libs postgresql-devel
    pip install pug

Bleeding Edge
~~~~~~~~~~~~~

Even the releases are very unstable, but if you want to have the latest,
most broken code

::

    pip install git+git://github.com/totalgoodlane/pug.git@master

Warning
~~~~~~~

This software is in alpha testing. Install at your own risk.

--------------

Development
-----------

I love merging PRs and adding contributors to the ``__authors__`` list:

::

    git clone https://github.com/totalgood/pug-nlp.git

.. |Build Status| image:: https://travis-ci.org/totalgood/pug-nlp.svg?branch=master
   :target: https://travis-ci.org/totalgood/pug-nlp
.. |Coverage Status| image:: https://coveralls.io/repos/totalgood/pug-nlp/badge.png
   :target: https://coveralls.io/r/totalgood/pug-nlp
.. |Version Status| image:: https://pypip.in/v/pug-nlp/badge.png
   :target: https://pypi.python.org/pypi/pug-nlp/
.. |Downloads| image:: https://pypip.in/d/pug-nlp/badge.png
   :target: https://pypi.python.org/pypi/pug-nlp/

