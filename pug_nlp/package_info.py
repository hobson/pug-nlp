# Explicitly declare some header/meta information (about `pug`)
#  rather than allowing python to populate header info automatically.
# This allows simple parsers (in setup.py) to extract them
#  without importing this file or the pug package or its __init__.py
import os
__namespace_package__ = 'pug'
__subpackage__ = 'nlp'
__url__ = "https://github.com/hobson/{}-{}".format(__namespace_package__, __subpackage__)
__version__ = '0.1.1'
__author__ = "Hobson <hobson@totalgood.com>"
__authors__ = (
    "Hobson <hobson@totalgood.com>",
    )
__package_path__ = os.path.abspath('.')


def try_read(filename, path=__package_path__):
    try:
        return open(filename, 'r').read()
    except:
        try:
            return open(os.path.join(path, '..', filename), 'r').read()
        except:
            try:
                return open(os.path.join(path, '..', '..', filename), 'r').read()
            except:
                return ''


__license__ = try_read('LICENSE.txt', __package_path__)
__doc__ = "{}.{} -- Natural Language Processing (NLP) python utilities by and for the PDX Python User Group (PUG)".format(
    __namespace_package__, __subpackage__)
__doc__ = try_read('README.md', __package_path__) or __doc__
