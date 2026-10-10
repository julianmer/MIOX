####################################################################################################
#                                              cli.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Console entry point for the ``miox`` command: ``info`` (what a file holds: data      #
#          shape, standard keys and their sources), ``sniff`` (format of files or directories) and #
#          ``audit`` (metadata coverage table of a file).                                          #
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
from miox.data import (
    FitResult,
    MRImage,
    NoiseCovariance,
    Phantom,
    PhysioData,
    PriorSet,
    Protocol,
    PulseSequence,
    RawData,
    Surface,
    Tractogram,
    Transform,
)
from miox.readers import Loader, ReaderError
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

        info = commands.add_parser("info", help="data shape, standard keys and their sources")
        info.add_argument("paths", nargs="+", type=Path)
        info.add_argument("--json", action="store_true", help="one JSON object per dataset")
        info.set_defaults(func=self.info)

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

    def info(self, args: argparse.Namespace) -> int:
        """Summary of every dataset in every path; status 1 if any path cannot be read."""
        loader = Loader()
        status = 0
        for path in args.paths:
            try:
                datasets = loader.load(path)
            except ReaderError as e:
                print(f"miox info: {e}", file=sys.stderr)
                status = 1
                continue
            for d in datasets:
                print(json.dumps(d.to_dict(), default=str) if args.json else self._describe(d))
        return status

    def _describe(self, d) -> str:
        lines = [f"{d.provenance.paths[0]}: {d.provenance.format} ({d.provenance.reader})"]
        if isinstance(d, RawData):
            trajectory = ", with trajectory" if d.trajectory is not None else ""
            waveforms = f", {len(d.waveforms)} waveforms" if d.waveforms else ""
            lines.append(
                f"  {d.name}: {d.n_readouts} readouts x {d.n_coils} coils x "
                f"{d.data.shape[-1]} samples, {d.data.dtype}{trajectory}{waveforms}"
            )
        elif isinstance(d, MRImage):
            dims = " ".join(f"{name}={n}" for name, n in zip(d.dims, d.shape, strict=True))
            sizes = " x ".join(f"{v:.4g}" for v in d.voxel_sizes)
            lines.append(f"  {d.name}: {dims}, {d.data.dtype}, voxels {sizes} mm")
        elif isinstance(d, PriorSet):
            lines.append(f"  {d.name}: {len(d.metabolites)} metabolites, {len(d.lines)} lines")
        elif isinstance(d, Protocol):
            where = f" in {' / '.join(d.location)}" if d.location else ""
            lines.append(f"  protocol {d.name}{where}")
        elif isinstance(d, Phantom):
            lines.append(f"  {d.name}: {d.n_spins} spins, {', '.join(d.properties)}")
        elif isinstance(d, Tractogram):
            lines.append(f"  {d.name}: {d.n_streamlines} streamlines, {d.points.shape[0]} "
                         f"points ({d.space})")
        elif isinstance(d, Surface):
            faces = f", {d.faces.shape[0]} faces" if d.faces is not None else ""
            data = ", ".join(list(d.vertex_data) + list(d.labels))
            lines.append(f"  {d.name}: {d.n_vertices} vertices{faces}{', ' + data if data else ''}")
        elif isinstance(d, Transform):
            lines.append(f"  {d.name}: {d.kind} transform ({d.convention})")
        elif isinstance(d, PhysioData):
            rate = f" at {d.sampling_frequency:g} Hz" if d.sampling_frequency else ""
            events = "".join(f", {len(v)} {k}" for k, v in d.events.items())
            channels = ", ".join(d.channels)
            lines.append(f"  {d.name}: {d.n_samples} samples of {channels}{rate}{events}")
        elif isinstance(d, NoiseCovariance):
            lines.append(
                f"  {d.name}: noise covariance of {d.n_coils} channels, {d.sample_count} samples"
            )
        elif isinstance(d, PulseSequence):
            events = ", ".join(f"{len(t)} {n}" for n, t in d.events.items())
            duration = f", {d.duration:.4g} s" if d.duration is not None else ""
            lines.append(f"  {d.name}: Pulseq {d.version}, {d.n_blocks} blocks{duration}; {events}")
        elif isinstance(d, FitResult):
            voxel = f", voxel {d.voxel}" if d.voxel else ""
            curves = f", curves: {', '.join(d.curves)}" if d.curves else ""
            lines.append(f"  {d.name}: {len(d.rows)} metabolites{voxel}{curves}")
            lines.extend(f"    {name}: {value}" for name, value in d.diagnostics.items())
        else:
            dims = " ".join(f"{name}={n}" for name, n in zip(d.dims, d.shape, strict=True))
            width = "" if d.spectral_width is None else f" ({d.spectral_width:.6g} Hz)"
            lines += [f"  {d.name}: {dims}", f"    {d.acquisition()}{width}, {d.n_points} points"]
        width = max((len(key) for key in d.meta), default=0)
        for key, value in d.meta.items():
            text = json.dumps(value, default=str)
            if len(text) > 50:
                text = text[:49] + "…"
            lines.append(f"    {key:<{width}}  {text:<50}  {d.meta.source[key]}")
        lines.append(f"    vendor header: {len(d.meta.vendor_fields())} fields")
        lines.extend(f"    note: {note}" for note in d.provenance.notes)
        return "\n".join(lines)

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
        """Coverage table as Markdown; status 2 if the file cannot be read."""
        try:
            print(Auditor().audit(args.path).to_markdown(), end="")
        except ReaderError as e:
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
