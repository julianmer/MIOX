####################################################################################################
#                                      test_brainvoyager.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: BrainVoyager readers on synthetic files: a VMR version 4 (gzipped too) with its axes    #
#          and framing cube, an SSM, a protocol with parametric weights, a design matrix, a VOI in #
#          TAL space, a POI, a transformation with an extra VMR matrix, and look-alikes.           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

import gzip
import struct

import numpy as np

# own
import miox


def vmr_bytes(data: np.ndarray, voxel=(1.0, 1.0, 1.0), offsets=(0, 0, 0), cube=8,
              convention=1) -> bytes:
    """A version 4 VMR of ``data`` given as (X, Y, Z)."""
    x, y, z = data.shape
    post = struct.pack("<4h", *offsets, cube) + struct.pack("<2i", 0, 0) + bytes(48)
    post += struct.pack("<2i", 0, 0) + bytes(16) + struct.pack("<i", 0)
    post += struct.pack("<2B", convention, 1) + struct.pack("<3f", *voxel)
    post += struct.pack("<2B", 1, 0) + struct.pack("<3i", 0, 0, 0)
    return (struct.pack("<4H", 4, x, y, z) + data.transpose(2, 1, 0).astype("u1").tobytes()
            + post)


def test_vmr_axes_and_cube(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    path = tmp_path / "a.vmr"
    path.write_bytes(vmr_bytes(data, voxel=(2.0, 1.0, 1.0), offsets=(1, 0, 0)))
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.BRAINVOYAGER_VMR, miox.Kind.IMAGING)
    [image] = miox.load(path)
    assert np.array_equal(image.data, data)
    # voxel (0, 0, 0): X = 1 (offset) of an 8 cube, centre 4: A = (4 - 1) * 2, S = 4, R = 4
    assert np.allclose(image.affine @ [0, 0, 0, 1], [4.0, 6.0, 4.0, 1.0])
    assert np.allclose(image.affine[:3, :3], [[0, 0, -1], [-2, 0, 0], [0, -1, 0]])


def test_vmr_neurological_and_gzipped(tmp_path):
    data = np.ones((2, 2, 2), dtype=np.uint8)
    path = tmp_path / "b.vmr.gz"
    path.write_bytes(gzip.compress(vmr_bytes(data, convention=2)))
    assert miox.sniff(path).format == miox.Format.BRAINVOYAGER_VMR
    [image] = miox.load(path)
    assert image.name == "b" and image.affine[0, 2] == 1.0


def test_ssm(tmp_path):
    path = tmp_path / "a.ssm"
    path.write_bytes(struct.pack("<hii", 2, 3, 10) + struct.pack("<3i", 9, 0, 4))
    [s] = miox.load(path)
    assert np.array_equal(s.vertex_data["target_vertex"], [9, 0, 4])


PRT = """FileVersion:        3

ResolutionOfTime:   msec

Experiment:         Test

ParametricWeights:  1

NrOfConditions:     2

rest
1
   0  1000  1.0
Color: 10 20 30

task one
2
 1000  2000  0.5
 3000  4000  2.5
Color: 1 2 3
"""


def test_protocol(tmp_path):
    path = tmp_path / "a.prt"
    path.write_text(PRT)
    sniff = miox.sniff(path)
    assert (sniff.format, sniff.kind) == (miox.Format.BRAINVOYAGER_PRT, miox.Kind.PROTOCOL)
    [p] = miox.load(path)
    task = p.meta.vendor["conditions"][1]
    assert p.name == "Test" and p.meta["ProtocolTimeResolution"] == "msec"
    assert task == {"name": "task one", "start": [1000.0, 3000.0], "stop": [2000.0, 4000.0],
                    "weight": [0.5, 2.5], "color": [1, 2, 3]}


def test_design_matrix(tmp_path):
    path = tmp_path / "a.sdm"
    path.write_text('FileVersion: 1\nNrOfPredictors: 2\nNrOfDataPoints: 3\nIncludesConstant: 1\n'
                    'FirstConfoundPredictor: 2\n\n255 0 0 0 0 255\n"Task" "Constant"\n'
                    "0.5 1\n-1.5e-3 1\n2 1\n")
    [p] = miox.load(path)
    assert np.allclose(p.meta.vendor["matrix"], [[0.5, 1], [-0.0015, 1], [2, 1]])
    assert [c["name"] for c in p.meta.vendor["predictors"]] == ["Task", "Constant"]


def test_voi_in_tal_space(tmp_path):
    path = tmp_path / "a.voi"
    path.write_text("FileVersion: 4\nReferenceSpace: TAL\nLeftRightConvention: 1\n"
                    "NrOfVOIs: 1\nNameOfVOI: spot\nColorOfVOI: 255 0 0\nNrOfVoxels: 2\n"
                    "10 -20 30\n11 -20 30\n")
    [image] = miox.load(path)
    assert image.data.shape == (1, 1, 2) and image.data.sum() == 2
    for k, x in enumerate((11, 10)):  # Z runs right to left: the larger x comes first
        assert np.allclose(image.affine @ [0, 0, k, 1], [x, -20, 30, 1])


def test_poi(tmp_path):
    path = tmp_path / "a.poi"
    path.write_text('FileVersion: 2\nFromMeshFile: "m.srf"\nNrOfMeshVertices: 5\nNrOfPOIs: 1\n'
                    'NameOfPOI: "patch"\nInfoTextFile: ""\nColorOfPOI: 255 0 0\nLabelVertex: 1\n'
                    "NrOfVertices: 2\n1\n3\n")
    [s] = miox.load(path)
    assert np.array_equal(s.labels["patch"]["values"], [0, 1, 0, 1, 0])


def test_transformation_with_extra_vmr_matrix(tmp_path):
    path = tmp_path / "a.trf"
    rows = "\n".join(" ".join(map(str, r)) for r in np.eye(4)) + "\n"
    path.write_text("FileVersion: 7\nDataFormat: Matrix\n" + rows + "TransformationType: 1\n"
                    "ExtraVMRTransf: 1\n" + rows + 'SourceFile: "a.fmr"\nTargetFile: "b.vmr"\n')
    transforms = miox.load(path)
    assert [t.name for t in transforms] == ["a", "a_extra_vmr"]
    assert transforms[0].source == {"file": "a.fmr"} and transforms[0].ras_matrix is None


def test_lookalikes(tmp_path):
    prt = tmp_path / "notes.prt"
    prt.write_text("conditions: none\n")
    vmr = tmp_path / "short.vmr"
    vmr.write_bytes(struct.pack("<4H", 4, 100, 100, 100) + bytes(100))
    for path in (prt, vmr):
        assert miox.sniff(path).format == miox.Format.UNKNOWN
