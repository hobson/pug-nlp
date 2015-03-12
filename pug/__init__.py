from pkg_resources import declare_namespace
declare_namespace(__name__)

import nlp
import data
__all__ = ['nlp', 'data']