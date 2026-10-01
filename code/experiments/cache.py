"""Size-bounded LRU storage for deterministic benchmark intermediates."""
from collections import OrderedDict
from collections.abc import MutableMapping
import sys
import numpy as np


def storage_bytes(value):
    if isinstance(value, np.ndarray):
        base = value
        while isinstance(base.base, np.ndarray):
            base = base.base
        return max(sys.getsizeof(value), sys.getsizeof(base), base.nbytes)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(storage_bytes(k) + storage_bytes(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value) + sum(storage_bytes(v) for v in value)
    return sys.getsizeof(value)


class ByteLRU(MutableMapping):
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.bytes_used = 0
        self._items = OrderedDict()

    def __getitem__(self, key):
        value, size = self._items[key]
        self._items.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self._items:
            del self[key]
        size = storage_bytes(key) + storage_bytes(value)
        if size > self.max_bytes:
            return
        while self.bytes_used + size > self.max_bytes and self._items:
            del self[next(iter(self._items))]
        self._items[key] = (value, size)
        self.bytes_used += size

    def __delitem__(self, key):
        _, size = self._items.pop(key)
        self.bytes_used -= size

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)
