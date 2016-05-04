"""Keep this as simple as possible to minimize the possability of error when used within a django settings.py file"""

import sys
import os

def get(var_name, default=False, verbosity=0):
    """ Get the environment variable or assume a default, but let the user know about the error."""
    try:
        value = os.environ[var_name]
        if str(value).strip().lower() in ['false', 'no', 'off' '0', 'none', 'null']:
            return None
        return value
    except:
        if verbosity >= 0:
            msg = "Unable to find the %s environment variable.\nUsing the value %s (the default) instead.\n" % (var_name, default)
            if verbosity > 0:
                from traceback import format_exc
                sys.stderr.write(format_exc())
            sys.stderr.write(msg)
        return default