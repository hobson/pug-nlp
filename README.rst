pug-nlp |alt text|
==================

Natural Language Processing (NLP) algorithms
--------------------------------------------

This is a namespace package (sub-package) of the pug package a
collection of utilities by and for the PDX Python User Group.

--------------

Introduction
------------

Natural Language Processing (NLP) algorithms by and for the PDX Python
User Group (PUG).

--------------

Installation
------------

On a Posix System
~~~~~~~~~~~~~~~~~

You really want to contribute, right?

::

    git clone https://github.com/hobson/pug-nlp.git

If your a user and not a developer, and have an up-to-date posix OS with
the postgres, xml2, and xlst development packages installed, then just
use ``pip``.

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

    pip install git+git://github.com/hobsonlane/pug.git@master

Warning
~~~~~~~

This software is in alpha testing. Install at your own risk.

--------------

Development
-----------

I love merging PRs and adding contributors to the **authors** list:

::

    git clone https://github.com/hobson/pug-nlp.git

.. |alt text| image:: https://travis-ci.org/hobson/pug-nlp.svg?branch=master
