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
# from (``meta.source["EchoTime"]``, readable; ``meta.fields["EchoTime"]``, the path of the vendor #
# header field when the value was read from one), and the vendor header kept whole                 #
# (``meta.vendor``).                                                                               #
#                                                                                                  #
#**************************************************************************************************#
class Meta(dict):
    def __init__(self, vendor: dict | None = None):
        super().__init__()
        self.source: dict[str, str] = {}
        self.fields: dict[str, str] = {}
        self.vendor: dict = vendor if vendor is not None else {}

    def set(self, key: str, value: Any, source: str, field: str | None = None) -> None:
        """Store ``value`` under ``key`` and record where it came from."""
        self[key] = value
        self.source[key] = source
        if field is not None:
            self.fields[key] = field

    def vendor_fields(self) -> dict[str, Any]:
        """Every leaf of the vendor header by its path, the parts joined with ``/``."""
        fields = {}
        stack = [("", self.vendor)] if self.vendor else []
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
