# Explicitly declare some header/meta information (about `pug`)
#  rather than allowing python to populate header info automatically.
# This allows simple parsers (in setup.py) to extract them
#  without importing this file or __init__.py
__namespace_package__ = 'pug'
__subpackage__ = 'ann'
__doc__ = "{}.{} -- Artificial Neural Netwwork (ANN) utilities by and for the PDX Python User Group (PUG)".format(__namespace_package__, __subpackage__)
__url__ = "https://github.com/hobson/{}-{}".format(__namespace_package__, __subpackage__)
__version__ = '0.0.1'
__author__ = "Hobson <hobson@totalgood.com>"
__authors__ = (
    "Hobson <hobson@totalgood.com>",
    )

import os
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
__doc__ = try_read('README.md', __package_path__) or __doc__

