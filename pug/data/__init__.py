from pkg_resources import declare_namespace
declare_namespace(__name__)

import tobes

__all__ = globals().get('__all__', []) + ['tobes']