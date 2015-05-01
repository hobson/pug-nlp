# pug-nlp

[![Build Status](https://travis-ci.org/hobson/pug-nlp.svg?branch=master "Travis Build & Test Status")](https://travis-ci.org/hobson/pug-nlp)
[![Coverage Status](https://coveralls.io/repos/hobson/pug-nlp/badge.png)](https://coveralls.io/r/hobson/pug-nlp)
[![Latest Release Version](https://badge.fury.io/py/pug-nlp.svg)](https://pypi.python.org/pypi/pug-nlp/)
<!-- [![Downloads](https://pypip.in/d/pug-nlp/badge.png)](https://pypi.python.org/pypi/pug-nlp/) -->

## PUG Natural Language Processing (NLP) Utilities

This sub-package of the pug namespace package, provides natural language processing (NLP) and text processing utilities built by and for the PDX Python User Group (PUG).

---

## Installation

### On a Posix System

You really want to contribute, right?

    git clone https://github.com/hobson/pug-nlp.git

If you're a user and not a developer, and have an up-to-date posix OS with the postgres, xml2, and xlst development packages installed, then just use `pip`.

    pip install pug-nlp

### Fedora

If you're on Fedora >= 16 but haven't done a lot of python binding development, then you'll need some libraries before pip will succeed.

    sudo yum install -y python-devel libxml2-devel libxslt-devel gcc-gfortran python-scikit-learn postgresql postgresql-server postgresql-libs postgresql-devel
    pip install pug

### Bleeding Edge

Even the releases are very unstable, but if you want to have the latest, most broken code

    pip install git+git://github.com/hobsonlane/pug.git@master

### Warning

This software is in alpha testing.  Install at your own risk.

---

## Development

I love merging PRs and adding contributors to the `__authors__` list:

    git clone https://github.com/hobson/pug-nlp.git


