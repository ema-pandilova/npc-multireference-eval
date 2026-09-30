"""Scanner-aware, inter-rater-calibrated segmentation uncertainty on NPC MRI.

Modules:
    paths      configuration loading and programmatic path verification
    discovery  patient / modality / slice discovery from the filesystem
    imaging    TIFF+NIfTI+DICOM metadata reading and image/mask geometry checks
    clinical   clinical CSV profiling and defensive imaging<->clinical ID matching
    manifest   patient-level manifest construction
    splits     leakage-free train/calibration/test splitting with hard guards
    overlays   image/mask and multi-rater overlay rendering
"""

__version__ = "0.1.0"
