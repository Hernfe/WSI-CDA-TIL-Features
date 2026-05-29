"""Manifest loading: resolve slide-id → (wsi_path, roi_geojson, ...) mappings.

Two input modes:
  1. Manifest CSV with columns: slide_id, wsi_path, roi_geojson
     (additional columns are passed through as metadata)
  2. Directory scan: slides_dir + roi_masks_dir with pattern substitution
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd


REQUIRED_COLUMNS = {"slide_id"}
OPTIONAL_COLUMNS = {"wsi_path", "roi_geojson", "tumor_grid"}


def load_manifest(csv_path: Path) -> pd.DataFrame:
    """Read and validate a manifest CSV.

    Required column: slide_id.
    Optional columns: wsi_path, roi_geojson, tumor_grid, cells_csv.
    Extra columns are preserved.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest {csv_path} missing required columns: {sorted(missing)}"
        )
    df["slide_id"] = df["slide_id"].str.strip()
    df = df[df["slide_id"] != ""].reset_index(drop=True)
    return df


def pattern_to_path(pattern: str, slide_id: str) -> Path:
    """Substitute {slide_id} placeholder in a file pattern and return a Path."""
    return Path(pattern.replace("{slide_id}", slide_id))


def resolve_path(base_dir: Path, pattern: str, slide_id: str) -> Path:
    """Resolve a file pattern relative to a base directory."""
    rel = pattern_to_path(pattern, slide_id)
    if rel.is_absolute():
        return rel
    return base_dir / rel


def discover_slides(
    slides_dir: Path,
    roi_masks_dir: Path,
    roi_pattern: str = "{slide_id}_selected_cdaroi_stage5b.geojson",
    slide_ext: str = ".mrxs",
) -> pd.DataFrame:
    """Scan slides_dir for WSIs and match them to ROI masks by slide_id.

    slide_id is derived by stripping the extension from the WSI filename.
    Only slides with a matching ROI mask are returned.
    """
    rows = []
    ext = slide_ext if slide_ext.startswith(".") else "." + slide_ext
    for wsi_path in sorted(slides_dir.glob(f"*{ext}")):
        slide_id = wsi_path.stem
        roi_path = resolve_path(roi_masks_dir, roi_pattern, slide_id)
        if roi_path.exists():
            rows.append({"slide_id": slide_id, "wsi_path": str(wsi_path),
                         "roi_geojson": str(roi_path)})
    if not rows:
        return pd.DataFrame(columns=["slide_id", "wsi_path", "roi_geojson"])
    return pd.DataFrame(rows)


def iter_rows(df: pd.DataFrame) -> Iterator[dict]:
    """Iterate manifest rows as dicts."""
    for _, row in df.iterrows():
        yield row.to_dict()


_PRIVATE_PATTERNS = re.compile(
    r"/mnt/files|/data/melanoma|/mnt/melanoma|Blackstorm|FIMM|HUS"
    r"|melanoma_selected_slides|final_features|qc_exports",
    re.IGNORECASE,
)


def validate_no_private_paths(df: pd.DataFrame) -> None:
    """Raise if any path column contains known private path fragments."""
    for col in df.columns:
        for val in df[col].astype(str):
            if _PRIVATE_PATTERNS.search(val):
                raise ValueError(
                    f"Manifest column '{col}' contains a private path: {val!r}. "
                    "Remove or anonymise before sharing."
                )
