####################################################################################################
#                                             keys.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 01/10/26                                                                                #
#                                                                                                  #
# Purpose: Key table: the standard metadata vocabulary. The NIfTI-MRS required and standard keys   #
#          are taken from the nifti-mrs package so the two never drift, plus the keys the standard #
#          does not define. Each entry carries the unit its value is stored in, the MRS-BIDS       #
#          sidecar name where one exists, and its sheet name and unit.                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from __future__ import annotations

from dataclasses import dataclass

from nifti_mrs.definitions import dimension_tags, nifti_mrs_version, required, standard_defined


#**************************************************************************************************#
#                                               Key                                                #
#**************************************************************************************************#
#                                                                                                  #
# One standard metadata key: its unit and type, where it comes from, and its names in MRS-BIDS and #
# in the sheet view.                                                                               #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Key:
    name: str
    unit: str | None
    """Unit the value is stored in under ``meta[name]`` (SI where the standard says so)."""
    type: tuple
    """Python types accepted, as in ``nifti_mrs.definitions`` (``(list, float)`` for arrays)."""
    doc: str
    origin: str
    """``nifti-mrs:required``, ``nifti-mrs:standard`` or ``miox``."""
    anon: bool = False
    """Removed by anonymisation."""
    bids: str | None = None
    """MRS-BIDS sidecar field name (BIDS schema ``rules/sidecars/mrs.yaml``)."""
    sheet: str | None = None
    """Short name in the sheet view."""
    sheet_unit: str | None = None
    sheet_factor: float | None = None
    """``sheet value = meta value * sheet_factor``; ``None`` means a derivation rule is needed."""


#**************************************************************************************************#
#                                             KeyTable                                             #
#**************************************************************************************************#
#                                                                                                  #
# All standard keys by name: the NIfTI-MRS required and standard keys, then the keys the standard  #
# does not define. Also holds the NIfTI-MRS version and dimension tags it was built from.          #
#                                                                                                  #
#**************************************************************************************************#
class KeyTable(dict):
    # NIfTI-MRS key -> MRS-BIDS sidecar name, only names in the BIDS schema
    BIDS = {
        "SpectrometerFrequency": "SpectrometerFrequency",
        "ResonantNucleus": "ResonantNucleus",
        "SpectralWidth": "SpectralWidth",
        "EchoTime": "EchoTime",
        "MixingTime": "MixingTime",
        "ExcitationFlipAngle": "FlipAngle",
        "WaterSuppressed": "WaterSuppression",
        "WaterSuppressionType": "WaterSuppressionTechnique",
        "Manufacturer": "Manufacturer",
        "ManufacturersModelName": "ManufacturersModelName",
        "DeviceSerialNumber": "DeviceSerialNumber",
        "SoftwareVersions": "SoftwareVersions",
        "InstitutionName": "InstitutionName",
        "InstitutionAddress": "InstitutionAddress",
        "RxCoil": "ReceiveCoilName",
        "SequenceName": "SequenceName",
        "EditCondition": "EditCondition",
        "EditPulse": "EditPulse",
    }

    # NIfTI-MRS key -> sheet name, unit and factor (sheet times are in ms, the standard stores s)
    SHEET = {
        "EchoTime": ("TE", "ms", 1e3),
        "MixingTime": ("TM", "ms", 1e3),
        "SpectralWidth": ("Bandwidth", "Hz", 1.0),
        "ResonantNucleus": ("Nucleus", None, 1.0),
    }

    # keys the NIfTI-MRS standard does not define
    MIOX = (
        Key(
            "FieldStrength",
            "T",
            (float,),
            "Static field strength, derived from SpectrometerFrequency and the gyromagnetic ratio "
            "of ResonantNucleus unless the file states it.",
            "miox",
            bids="MagneticFieldStrength",
            sheet="Bfield",
            sheet_unit="T",
            sheet_factor=1.0,
        ),
        Key(
            "DwellTime",
            "s",
            (float,),
            "Time between consecutive complex samples of the FID; inverse of SpectralWidth.",
            "miox",
        ),
        Key(
            "DiffusionBValue",
            "s/mm^2",
            (list, float),
            "Diffusion weighting of each volume of an image.",
            "miox",
        ),
        Key(
            "DiffusionGradientOrientation",
            None,
            (list, float),
            "Diffusion gradient direction of each volume as a unit vector in RAS world "
            "coordinates; zero where the volume is not diffusion weighted.",
            "miox",
        ),
        Key(
            "NumberOfPoints",
            None,
            (int,),
            "Number of complex samples per FID.",
            "miox",
            bids="NumberOfSpectralPoints",
            sheet="Samples",
            sheet_unit=None,
            sheet_factor=1.0,
        ),
        Key(
            "ReferencePPM",
            "ppm",
            (float,),
            "Chemical shift the file assigns to 0 Hz in the data (SpectrometerFrequency): a signal "
            "at f Hz lies at ReferencePPM - f / SpectrometerFrequency.",
            "miox",
        ),
        Key(
            "NumberOfAverages",
            None,
            (int,),
            "Number of acquired averages (transients) per spectrum, before any combination.",
            "miox",
        ),
        Key(
            "H1ResonanceFrequency",
            "MHz",
            (float,),
            "Proton resonance frequency of the system, whatever nucleus the data was acquired on.",
            "miox",
        ),
        Key(
            "PatientPosition",
            None,
            (str,),
            "Position of the patient in the scanner as a DICOM code (HFS, HFP, FFS, ...).",
            "miox",
        ),
        Key(
            "NumberOfReceiveChannels",
            None,
            (int,),
            "Receive channels of the system available for the acquisition.",
            "miox",
        ),
        Key(
            "EncodedMatrixSize",
            None,
            (list, int),
            "Matrix size of the encoded k-space (x, y, z), oversampling included.",
            "miox",
        ),
        Key(
            "EncodedFieldOfView",
            "mm",
            (list, float),
            "Field of view of the encoded k-space (x, y, z).",
            "miox",
        ),
        Key(
            "ReconMatrixSize",
            None,
            (list, int),
            "Matrix size of the reconstructed image (x, y, z).",
            "miox",
        ),
        Key(
            "ReconFieldOfView",
            "mm",
            (list, float),
            "Field of view of the reconstructed image (x, y, z).",
            "miox",
        ),
        Key(
            "EncodingLimits",
            None,
            (dict,),
            "Per encoding counter (kspace_encoding_step_1, slice, repetition, ...) its minimum, "
            "maximum and centre index.",
            "miox",
        ),
        Key(
            "KSpaceTrajectory",
            None,
            (str,),
            "Sampling trajectory of the readouts: cartesian, epi, radial, goldenangle, spiral or "
            "other.",
            "miox",
        ),
    )

    # BIDS MRI sidecar fields without a NIfTI-MRS or MIOX counterpart: name, unit, types, meaning
    BIDS_MRI = (
        ("Modality", None, (str,), "Imaging modality, e.g. MR."),
        ("SeriesDescription", None, (str,), "Free-text description of the series."),
        ("SeriesNumber", None, (int,), "Number of the series in the session."),
        ("AcquisitionNumber", None, (int,), "Number of the acquisition within the series."),
        ("AcquisitionTime", None, (str,), "Time of day the acquisition started, hh:mm:ss.ffffff."),
        ("ImageType", None, (list, str), "DICOM image type values (e.g. ORIGINAL, PRIMARY, M)."),
        ("StudyDescription", None, (str,), "Free-text description of the study."),
        ("ProcedureStepDescription", None, (str,), "Description of the procedure step."),
        ("StationName", None, (str,), "Name of the scanner station."),
        ("InstitutionalDepartmentName", None, (str,), "Department of the institution."),
        ("BodyPart", None, (str,), "Body part examined."),
        ("SliceThickness", "mm", (float,), "Nominal slice thickness."),
        ("SpacingBetweenSlices", "mm", (float,), "Distance between centres of adjacent slices."),
        ("MRAcquisitionType", None, (str,), "2D or 3D acquisition."),
        ("ScanningSequence", None, (str,), "DICOM scanning sequence (e.g. SE, GR, EP)."),
        ("SequenceVariant", None, (str,), "DICOM sequence variant (e.g. SK, SP)."),
        ("ScanOptions", None, (str,), "DICOM scan options (e.g. FS, PFP)."),
        ("PulseSequenceType", None, (str,), "General type of the pulse sequence."),
        ("PixelBandwidth", "Hz", (float,), "Receiver bandwidth per pixel."),
        ("PercentPhaseFOV", "%", (float,), "Phase field of view relative to the readout one."),
        ("PercentSampling", "%", (float,), "Fraction of acquisition matrix lines sampled."),
        ("AcquisitionMatrixPE", None, (int,), "Acquired phase-encoding lines."),
        ("ReconMatrixPE", None, (int,), "Reconstructed phase-encoding lines."),
        ("PhaseEncodingSteps", None, (int,), "Number of phase-encoding steps."),
        ("PhaseEncodingDirection", None, (str,), "Phase-encoding axis and sign (i, j, k, -)."),
        ("InPlanePhaseEncodingDirectionDICOM", None, (str,), "ROW or COL, as DICOM states it."),
        ("ParallelReductionFactorInPlane", None, (float,), "In-plane acceleration factor."),
        ("ParallelReductionFactorOutOfPlane", None, (float,), "Slice acceleration factor."),
        ("MultibandAccelerationFactor", None, (float,), "Simultaneous multi-slice factor."),
        ("PartialFourier", None, (float,), "Fraction of k-space sampled (partial Fourier)."),
        ("EchoTrainLength", None, (int,), "Echoes per excitation."),
        ("EchoNumber", None, (int,), "Index of the echo in a multi-echo acquisition."),
        ("EffectiveEchoSpacing", "s", (float,), "Effective echo spacing (sampling interval)."),
        ("TotalReadoutTime", "s", (float,), "Readout time used for distortion correction."),
        ("SliceTiming", "s", (list, float), "Acquisition time of each slice from volume start."),
        ("DelayTime", "s", (float,), "Pause after each volume."),
        ("AcquisitionDuration", "s", (float,), "Duration of the acquisition."),
        ("SAR", "W/kg", (float,), "Specific absorption rate."),
        ("TablePosition", "mm", (list, float), "Table position."),
        ("ShimSetting", None, (list, float), "Shim currents, vendor units."),
        ("CoilCombinationMethod", None, (str,), "How the coil channels were combined."),
        ("MatrixCoilMode", None, (str,), "Coil acceleration mode (e.g. GRAPPA)."),
        ("NonlinearGradientCorrection", None, (bool,), "Gradient non-linearity corrected."),
        ("MTState", None, (bool,), "Magnetisation transfer pulse used."),
        ("SpoilingState", None, (bool,), "Spoiling used."),
        ("ImageOrientationPatientDICOM", None, (list, float), "DICOM ImageOrientationPatient."),
    )
    # fields dcm2niix writes beyond BIDS: name, unit, types, meaning
    DCM2NIIX = (
        ("AcquisitionDateTime", None, (str,), "Date and time the acquisition started, ISO 8601."),
        ("BaseResolution", None, (int,), "Siemens readout matrix size."),
        ("TxRefAmp", "V", (float,), "Siemens transmitter reference amplitude."),
        ("PhaseResolution", None, (float,), "Siemens phase resolution relative to readout."),
        ("PhaseOversampling", None, (float,), "Siemens phase oversampling fraction."),
        ("VendorReportedEchoSpacing", "s", (float,), "Echo spacing as the vendor reports it."),
        ("DerivedVendorReportedEchoSpacing", "s", (float,),
         "Echo spacing from the bandwidth per pixel, acceleration and oversampling."),
        ("BandwidthPerPixelPhaseEncode", "Hz", (float,), "Bandwidth per pixel, phase encoding."),
        ("RefLinesPE", None, (int,), "Siemens reference lines for parallel imaging."),
        ("Interpolation2D", None, (int,), "Siemens in-plane interpolation switch."),
        ("ConsistencyInfo", None, (str,), "Siemens software baseline of the protocol."),
        ("PulseSequenceDetails", None, (str,), "Siemens sequence file name."),
        ("WipMemBlock", None, (str,), "Siemens work-in-progress sequence free text."),
        ("CoilString", None, (str,), "Receive coil name when it differs from the elements."),
        ("ReceiveCoilActiveElements", None, (str,), "Receive coil elements used."),
        ("ImageOrientationText", None, (str,), "Siemens slice orientation text (Tra, Sag>Cor)."),
        ("ImageTypeText", None, (list, str), "Siemens XA image type text values."),
        ("PhaseEncodingAxis", None, (str,), "Phase-encoding axis (i, j) when the sign is unknown."),
        ("PhaseEncodingDirectionDisplayed", None, (str,), "UIH phase-encoding direction text."),
        ("PhaseEncodingPolarityGE", None, (str,), "GE phase-encoding polarity, Flipped/Unflipped."),
        ("PrescanReuseString", None, (str,), "GE prescan reuse settings."),
        ("InternalPulseSequenceName", None, (str,), "GE internal pulse sequence name."),
        ("PulseSequenceName", None, (str,), "Pulse sequence name when SequenceName is stated too."),
        ("NumberOfKSpaceTrajectories", None, (int,), "Number of k-space trajectories."),
        ("FrequencyEncodingSteps", None, (int,), "Frequency-encoding steps."),
        ("PhaseEncodingStepsOutOfPlane", None, (int,), "Out-of-plane phase-encoding steps."),
        ("PartialFourierEnabled", None, (str,), "Philips: partial matrix scanned (YES)."),
        ("PhaseEncodingStepsNoPartialFourier", None, (int,),
         "Phase-encoding steps before partial Fourier (Philips)."),
        ("ParallelAcquisitionTechnique", None, (str,), "Parallel imaging technique (SENSE, ...)."),
        ("PartialFourierDirection", None, (str,), "Partial Fourier direction."),
        ("SpoilingType", None, (str,), "Spoiling type (RF, GRADIENT, COMBINED)."),
        ("DiffusionScheme", None, (str,), "Siemens diffusion scheme (Bipolar, Monopolar)."),
        ("WaterFatShift", "pixels", (float,), "Philips water-fat shift."),
        ("EstimatedEffectiveEchoSpacing", "s", (float,),
         "Philips effective echo spacing estimated from the water-fat shift."),
        ("EstimatedTotalReadoutTime", "s", (float,),
         "Philips readout time estimated from the water-fat shift."),
        ("PhilipsRescaleSlope", None, (float,), "Philips RescaleSlope of the stored values."),
        ("PhilipsRescaleIntercept", None, (float,), "Philips RescaleIntercept of stored values."),
        ("PhilipsScaleSlope", None, (float,), "Philips private scale slope (2005,100E)."),
        ("UsePhilipsFloatNotDisplayScaling", None, (int,),
         "1 when values are floating point (stored / (RescaleSlope * ScaleSlope))."),
        ("PhilipsRWVSlope", None, (float,), "Philips real-world value slope."),
        ("PhilipsRWVIntercept", None, (float,), "Philips real-world value intercept."),
        ("Units", None, (str,), "Unit of the voxel values (e.g. Hz for a field map)."),
        ("ImageComments", None, (str,), "Free-text image comments."),
    )

    def __init__(self):
        super().__init__()
        self.nifti_mrs_version = tuple(nifti_mrs_version)
        self.dim_tags = tuple(dimension_tags)
        self._add_nifti_mrs(required, "nifti-mrs:required")
        self._add_nifti_mrs(standard_defined, "nifti-mrs:standard")
        self.update({key.name: key for key in self.MIOX})
        for name, unit, kind, doc in self.BIDS_MRI:
            self[name] = Key(name, unit, kind, doc, "bids", bids=name)
        for name, unit, kind, doc in self.DCM2NIIX:
            self[name] = Key(name, unit, kind, doc, "dcm2niix", bids=name)

    def _add_nifti_mrs(self, definitions: dict, origin: str) -> None:
        for name, definition in definitions.items():
            sheet, sheet_unit, sheet_factor = self.SHEET.get(name, (None, None, None))
            self[name] = Key(
                name=name,
                unit=definition.units,
                type=definition.type,
                doc=definition.doc,
                origin=origin,
                anon=definition.anon,
                bids=self.BIDS.get(name),
                sheet=sheet,
                sheet_unit=sheet_unit,
                sheet_factor=sheet_factor,
            )
