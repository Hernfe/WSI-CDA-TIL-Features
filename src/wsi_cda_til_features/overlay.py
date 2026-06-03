"""Map CDA cell centroids onto tumor patch grids and compute region summaries.

Cell assignment logic:
  Core counts  — CDA cells assigned to the selected tumor patch grid
                 (patch-level assignment via coordinate lookup).
  Total counts — all immune + tumor-cell rows in the cells CSV, which was
                 produced by running CDA on the peritumor ROI.
  Ring counts  — ring_immune = total_immune - core_immune
                 ring_tumor_cell = total_tumor_cell - core_tumor_cell

Geometry (loaded from core.geojson / peritumor.geojson) is used only for
area calculations.  The ring area is computed as peritumor.difference(core).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------------------
# Grid loading
# ---------------------------------------------------------------------------

_REQUIRED_GRID_COLS = {"x", "y", "predicted_class", "confidence", "is_tumor"}


def load_tumor_grid(path: Path) -> pd.DataFrame:
    """Load a tumor-grid file (parquet or CSV)."""
    if not path.exists():
        raise FileNotFoundError(f"Tumor grid not found: {path}")
    if path.suffix == ".parquet":
        grid = pd.read_parquet(path)
    else:
        grid = pd.read_csv(path)
    grid = grid.reset_index(drop=True)
    missing = _REQUIRED_GRID_COLS - set(grid.columns)
    if missing:
        raise ValueError(f"Tumor grid {path} missing columns: {sorted(missing)}")
    grid["x"] = grid["x"].astype(int)
    grid["y"] = grid["y"].astype(int)
    return grid


# ---------------------------------------------------------------------------
# Cell-to-patch assignment
# ---------------------------------------------------------------------------

_CELL_COUNT_COLUMNS = ["cda_immune_count", "cda_tumor_cell_count"]


def assign_cells_to_patches(
    cells: pd.DataFrame,
    patches: pd.DataFrame,
    patch_size: int,
) -> pd.DataFrame:
    """Assign each cell centroid to a patch by coordinate lookup.

    Returns a DataFrame with columns [patch_idx, cell_class].
    """
    xs = np.sort(patches["x"].unique())
    ys = np.sort(patches["y"].unique())
    patch_lookup: dict[tuple[int, int], int] = {
        (int(x), int(y)): i
        for i, (x, y) in enumerate(
            zip(patches["x"].astype(int), patches["y"].astype(int))
        )
    }

    rows: list[dict] = []
    missed = 0

    for row in cells.itertuples(index=False):
        xi = int(np.searchsorted(xs, row.cell_x, side="right")) - 1
        yi = int(np.searchsorted(ys, row.cell_y, side="right")) - 1

        if xi < 0 or yi < 0:
            missed += 1
            continue

        px, py = int(xs[xi]), int(ys[yi])
        if not (px <= row.cell_x < px + patch_size and py <= row.cell_y < py + patch_size):
            missed += 1
            continue

        patch_idx = patch_lookup.get((px, py))
        if patch_idx is None:
            missed += 1
            continue

        rows.append({"patch_idx": patch_idx, "cell_class": row.cell_class})

    if missed:
        print(f"  cells outside patch grid (missed): {missed}", file=sys.stderr)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["patch_idx", "cell_class"])


# ---------------------------------------------------------------------------
# Peritumor patch marking (KDTree — kept for QC and backward compatibility)
# ---------------------------------------------------------------------------


def mark_peritumor_patches(df: pd.DataFrame, radius_px: int) -> pd.Series:
    """Mark non-tumor patches within radius_px of any tumor patch centre.

    Note: this KDTree-based column is for QC and visualisation.  Peritumoral
    immune-cell counts for feature extraction are computed from the region
    summary (total counts minus core counts), not from this column.
    """
    false_series = pd.Series(False, index=df.index)
    if radius_px <= 0 or not df["is_tumor"].any():
        return false_series

    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        print(
            "  scikit-learn not available; is_peritumor_patch set to False",
            file=sys.stderr,
        )
        return false_series

    half = cfg.PATCH_SIZE / 2
    centers = np.column_stack([
        df["x"].to_numpy(dtype=float) + half,
        df["y"].to_numpy(dtype=float) + half,
    ])
    tumor_centers = centers[df["is_tumor"].to_numpy(dtype=bool)]
    tree = KDTree(tumor_centers)
    dist, _ = tree.query(centers, k=1)
    close = dist[:, 0] <= radius_px
    return pd.Series(close & ~df["is_tumor"].to_numpy(dtype=bool), index=df.index)


# ---------------------------------------------------------------------------
# Geometry area helpers
# ---------------------------------------------------------------------------


def _geom_area_mm2(geom, pixel_size_microns: float) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.area) * (pixel_size_microns / 1000.0) ** 2


# ---------------------------------------------------------------------------
# Main overlay function
# ---------------------------------------------------------------------------


def run_overlay(
    slide_id: str,
    tumor_grid_path: Path,
    cells_path: Path,
    output_path: Path,
    wsi_path: Path | None = None,
    patch_size: int = cfg.PATCH_SIZE,
    pixel_size_microns: float = cfg.PIXEL_SIZE_MICRONS,
    dilation_radius_um: float = cfg.DILATION_RADIUS_UM,
    use_geojson_fallback: bool = False,
    core_geojson_path: Path | None = None,
    peritumor_geojson_path: Path | None = None,
) -> pd.DataFrame:
    """Overlay CDA cells onto the tumor grid, compute region summary, write output.

    Parameters
    ----------
    slide_id:
        Slide identifier for logging.
    tumor_grid_path:
        Path to tumor-grid parquet or CSV.
    cells_path:
        Path to cells CSV (or detection GeoJSON if use_geojson_fallback).
    output_path:
        Where to write the overlay parquet.
    wsi_path:
        Required only when use_geojson_fallback=True (for offset reading).
    patch_size:
        Patch edge length in pixels at level 0.
    pixel_size_microns:
        Pixel size in microns at level 0.
    dilation_radius_um:
        Peritumor band radius in microns (used for KDTree QC column only when
        core_geojson_path / peritumor_geojson_path are not supplied).
    use_geojson_fallback:
        If True, load cells from GeoJSON instead of CSV.
    core_geojson_path:
        Path to {slide_id}_core.geojson for geometry-based area calculations.
    peritumor_geojson_path:
        Path to {slide_id}_peritumor.geojson.  CDA should have been run on
        this ROI; total cell counts are taken from the cells CSV.
    """
    from .cda_io import load_cells_csv, load_cells_geojson, read_wsi_offsets
    from .roi import load_core_geojson, load_peritumor_geojson

    print(f"=== overlay: {slide_id} ===")

    grid = load_tumor_grid(tumor_grid_path)
    grid["patch_idx"] = np.arange(len(grid), dtype=np.int64)
    for col in _CELL_COUNT_COLUMNS:
        grid[col] = 0

    if use_geojson_fallback:
        x_off, y_off = (0, 0)
        if wsi_path is not None:
            x_off, y_off = read_wsi_offsets(wsi_path)
            print(f"  WSI offsets: X={x_off}, Y={y_off}")
        cells = load_cells_geojson(cells_path, x_offset=x_off, y_offset=y_off)
        cells["cell_x"] = cells["cell_x"] + x_off
        cells["cell_y"] = cells["cell_y"] + y_off
    else:
        cells = load_cells_csv(cells_path)

    print(f"  cells loaded: {len(cells)}")

    # Total counts: all cells in the peritumor ROI (from the cells CSV/GeoJSON).
    total_immune_count = int((cells["cell_class"] == "immune").sum())
    total_tumor_cell_count = int((cells["cell_class"] == "tumor_cell").sum())

    # Core counts: cells assigned to tumor patch grid patches.
    assigned = assign_cells_to_patches(cells, grid, patch_size)

    if not assigned.empty:
        counts = (
            assigned
            .value_counts(["patch_idx", "cell_class"])
            .rename("n")
            .reset_index()
        )
        class_to_col = {
            "immune":     "cda_immune_count",
            "tumor_cell": "cda_tumor_cell_count",
        }
        for cell_class, col in class_to_col.items():
            sub = counts[counts["cell_class"] == cell_class][["patch_idx", "n"]]
            if not sub.empty:
                grid.loc[sub["patch_idx"].to_numpy(), col] = sub["n"].to_numpy()

    grid["cda_total_cells"] = grid[_CELL_COUNT_COLUMNS].sum(axis=1)

    patch_area = ((patch_size * pixel_size_microns) / 1000.0) ** 2
    grid["patch_area_mm2"] = patch_area
    grid["cda_immune_density_per_mm2"] = grid["cda_immune_count"] / patch_area
    grid["cda_tumor_cell_density_per_mm2"] = grid["cda_tumor_cell_count"] / patch_area

    # KDTree peritumor column (QC/visualisation — not used for primary counts).
    radius_px = cfg.um_to_px(dilation_radius_um, pixel_size_microns)
    grid["is_peritumor_patch"] = mark_peritumor_patches(grid, radius_px)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(output_path, index=False)

    # Region summary: core and ring counts + geometry-based areas.
    core_immune_count = int(grid["cda_immune_count"].sum())
    core_tumor_cell_count = int(grid["cda_tumor_cell_count"].sum())
    ring_immune_count = total_immune_count - core_immune_count
    ring_tumor_cell_count = total_tumor_cell_count - core_tumor_cell_count

    core_geom = load_core_geojson(core_geojson_path) if core_geojson_path else None
    peri_geom = load_peritumor_geojson(peritumor_geojson_path) if peritumor_geojson_path else None

    core_area_mm2 = _geom_area_mm2(core_geom, pixel_size_microns)
    peritumor_area_mm2 = _geom_area_mm2(peri_geom, pixel_size_microns)
    if peri_geom is not None and core_geom is not None:
        ring_area_mm2 = _geom_area_mm2(peri_geom.difference(core_geom), pixel_size_microns)
    else:
        ring_area_mm2 = peritumor_area_mm2 - core_area_mm2

    region_summary = {
        "slide_id":              slide_id,
        "core_immune_count":     core_immune_count,
        "core_tumor_cell_count": core_tumor_cell_count,
        "ring_immune_count":     ring_immune_count,
        "ring_tumor_cell_count": ring_tumor_cell_count,
        "unassigned_cell_count": 0,
        "core_area_mm2":         round(core_area_mm2, 6),
        "peritumor_area_mm2":    round(peritumor_area_mm2, 6),
        "ring_area_mm2":         round(ring_area_mm2, 6),
        "cda_roi_used":          str(peritumor_geojson_path) if peritumor_geojson_path else None,
    }

    summary_path = output_path.with_name(
        output_path.name.replace("_overlay.parquet", "_region_summary.json")
    )
    summary_path.write_text(json.dumps(region_summary, indent=2))

    print(f"  tumor patches:    {int(grid['is_tumor'].sum())}")
    print(f"  peritumor patches:{int(grid['is_peritumor_patch'].sum())}")
    print(f"  core immune cells:   {core_immune_count}")
    print(f"  ring immune cells:   {ring_immune_count}")
    print(f"  core area (mm2):  {core_area_mm2:.4f}")
    print(f"  ring area (mm2):  {ring_area_mm2:.4f}")
    print(f"  wrote: {output_path}")
    print(f"  wrote: {summary_path}")
    print(f"=== done: {slide_id} ===")
    return grid
