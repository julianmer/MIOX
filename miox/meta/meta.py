####################################################################################################
#                                             meta.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 02/10/26                                                                                #
#                                                                                                  #
# Purpose: The metadata of one dataset: standard keys with the source of every value, and the      #
#          complete vendor header as the parser returned it.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from typing import Any


#**************************************************************************************************#
#                                               Meta                                               #
#**************************************************************************************************#
#                                                                                                  #
# Standard keys by name (``meta["EchoTime"]``, units as in the key table), where each value came   #
# from (``meta.source["EchoTime"]``), and the vendor header kept whole (``meta.vendor``).          #
#                                                                                                  #
#**************************************************************************************************#
class Meta(dict):
    def __init__(self, vendor: dict | None = None):
        super().__init__()
        self.source: dict[str, str] = {}
        self.vendor: dict = vendor if vendor is not None else {}

    def set(self, key: str, value: Any, source: str) -> None:
        """Store ``value`` under ``key`` and record where it came from."""
        self[key] = value
        self.source[key] = source

    def vendor_fields(self) -> dict[str, Any]:
        """Every leaf of the vendor header by its path, the parts joined with ``/``."""
        fields = {}
        stack = [("", self.vendor)]
        while stack:
            prefix, node = stack.pop()
            if isinstance(node, dict) and node:
                children = list(node.items())
            elif isinstance(node, list) and node and any(isinstance(v, (dict, list)) for v in node):
                children = list(enumerate(node))
            else:
                fields[prefix] = node
                continue
            stack.extend((f"{prefix}/{k}" if prefix else str(k), v) for k, v in reversed(children))
        return fields
