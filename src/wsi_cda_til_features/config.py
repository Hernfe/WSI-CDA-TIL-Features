"""Runtime configuration: patch geometry, cell-class constants, thresholds.

All values can be overridden via environment variables or CLI args; nothing
here refers to any private path, cohort, or slide identifier.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Patch / pixel geometry
# ---------------------------------------------------------------------------

PATCH_SIZE: int = int(os.environ.get("WSI_PATCH_SIZE", "224"))
PIXEL_SIZE_MICRONS: float = float(os.environ.get("WSI_PIXEL_SIZE_MICRONS", "0.22"))

# ---------------------------------------------------------------------------
# Peritumor dilation
# ---------------------------------------------------------------------------

DILATION_RADIUS_UM: float = float(os.environ.get("WSI_DILATION_RADIUS_UM", "200"))

# ---------------------------------------------------------------------------
# DBSCAN section detection
# ---------------------------------------------------------------------------

SECTION_DBSCAN_EPS_UM: float = float(os.environ.get("WSI_SECTION_DBSCAN_EPS_UM", "3000"))
SECTION_DBSCAN_MIN_SAMPLES: int = int(os.environ.get("WSI_SECTION_DBSCAN_MIN_SAMPLES", "3"))

# ---------------------------------------------------------------------------
# TIL density threshold tiers (for fraction_tumor_patches_til_count_ge_N)
# ---------------------------------------------------------------------------

TIL_DENSITY_THRESHOLDS: list[int] = [5, 15, 30]

# ---------------------------------------------------------------------------
# Cell-class names
# ---------------------------------------------------------------------------

TIL_DETECTION_CLASS_NAMES: dict[str, str] = {
    "immune":     "Immune cells",
    "tumor_cell": "Tumor",
    "stroma":     "Stroma",
    "other":      "Other",
}

# Stroma and Other CDA classes are lower-confidence; excluded from default
# feature extraction unless explicitly requested.
LOW_TRUST_CLASSES: set[str] = {"stroma", "other"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def patch_area_mm2(patch_size: int = PATCH_SIZE,
                   pixel_size: float = PIXEL_SIZE_MICRONS) -> float:
    return ((patch_size * pixel_size) / 1000.0) ** 2


def um_to_px(microns: float, pixel_size: float = PIXEL_SIZE_MICRONS) -> int:
    return int(round(microns / pixel_size))
