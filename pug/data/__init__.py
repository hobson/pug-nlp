from pkg_resources import declare_namespace
declare_namespace(__name__)

import examples

__all__ = globals().get('__all__', []) + ['examples']