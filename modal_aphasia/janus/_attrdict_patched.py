# Monkey-patch collections module to fix attrdict import errors

import collections
import collections.abc

for type_name in ("Mapping", "MutableMapping", "Sequence"):
    setattr(collections, type_name, getattr(collections.abc, type_name))

from attrdict import AttrDict

__all__ = ["AttrDict"]
