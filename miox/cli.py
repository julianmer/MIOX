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
import json
import sys
from pathlib import Path

# own
from miox import __version__
from miox.audit import Auditor
from miox.sniff import Format, Sniffer


#**************************************************************************************************#
#                                               CLI                                                #
#**************************************************************************************************#
#                                                                                                  #
# The ``miox`` command line: one method per subcommand, each returning the exit status.         #
#                                                                                                  #
#**************************************************************************************************#
class CLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog="miox", description="Open any MR file.")
        self.parser.add_argument("--version", action="version", version=f"miox {__version__}")
        commands = self.parser.add_subparsers(dest="command", required=True)

        sniff = commands.add_parser("sniff", help="identify the format of files or directories")
        sniff.add_argument("paths", nargs="+", type=Path)
        sniff.add_argument("--json", action="store_true", help="one JSON object per line")
        sniff.set_defaults(func=self.sniff)

        audit = commands.add_parser("audit", help="metadata coverage table for a file (Markdown)")
        audit.add_argument("path", type=Path)
        audit.set_defaults(func=self.audit)

    def run(self, argv: list[str] | None = None) -> int:
        args = self.parser.parse_args(argv)
        return args.func(args)

    def sniff(self, args: argparse.Namespace) -> int:
        """One line (or JSON object) per path; status 1 if any path is unknown."""
        sniffer = Sniffer()
        status = 0
        for path in args.paths:
            result = sniffer.sniff(path)
            if result.format == Format.UNKNOWN:
                status = 1
            print(json.dumps(result.to_dict()) if args.json else result)
        return status

    def audit(self, args: argparse.Namespace) -> int:
        """Coverage table as Markdown; status 2 if the format cannot be audited."""
        try:
            print(Auditor().audit(args.path).to_markdown(), end="")
        except NotImplementedError as e:
            print(f"miox audit: {e}", file=sys.stderr)
            return 2
        return 0


#**************************************************************************************************#
#                                               main                                               #
#**************************************************************************************************#
#                                                                                                  #
# Console-script entry point.                                                                      #
#                                                                                                  #
#**************************************************************************************************#
def main(argv: list[str] | None = None) -> int:
    return CLI().run(argv)


if __name__ == "__main__":
    sys.exit(main())
