####################################################################################################
#                                              cli.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Console entry point for the ``miox`` command: ``sniff`` (format of files or          #
#          directories, text or JSON lines) and ``audit`` (metadata coverage table of a NIfTI-MRS  #
#          file).                                                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# own
from miox import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="miox", description="Open any MR file.")
    parser.add_argument("--version", action="version", version=f"miox {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sniff = sub.add_parser("sniff", help="identify the format of files or directories")
    p_sniff.add_argument("paths", nargs="+", type=Path)
    p_sniff.add_argument("--json", action="store_true", help="one JSON object per line")
    p_sniff.set_defaults(func=_sniff)

    p_audit = sub.add_parser("audit", help="metadata coverage table for a file (Markdown)")
    p_audit.add_argument("path", type=Path)
    p_audit.set_defaults(func=_audit)

    args = parser.parse_args(argv)
    return args.func(args)


def _sniff(args) -> int:
    from miox.sniff import UNKNOWN, sniff

    status = 0
    for path in args.paths:
        result = sniff(path)
        if result.format == UNKNOWN:
            status = 1
        if args.json:
            record = dataclasses.asdict(result)
            record["path"] = str(result.path)
            record["companions"] = [str(c) for c in result.companions]
            print(json.dumps(record))
        else:
            print(result)
    return status


def _audit(args) -> int:
    from miox.audit import audit

    try:
        print(audit(args.path).to_markdown(), end="")
    except NotImplementedError as e:
        print(f"miox audit: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
