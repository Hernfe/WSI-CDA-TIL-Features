"""Map CDA cell centroids onto tumor patch grids.

Adapted from the internal stage6_overlay.py (originally written for the
wsi-cda-til-features project). Generalised to work from file paths and
explicit config values rather than the private config module.
"""

from __future__ import annotations

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
# Peritumor marking
# ---------------------------------------------------------------------------


def mark_peritumor_patches(df: pd.DataFrame, radius_px: int) -> pd.Series:
    """Mark non-tumor patches within radius_px of any tumor patch centre."""
    false_series = pd.Series(False, index=df.index)
    if radius_px <= 0 or not df["is_tumor"].any():
        return false_series

    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        print(
            "  scikit-learn not available; peritumor labels set to False",
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
) -> pd.DataFrame:
    """Overlay CDA cells onto the tumor grid and write output_path.

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
        Peritumor band radius in microns.
    use_geojson_fallback:
        If True, load cells from GeoJSON instead of CSV.
    """
    from .cda_io import load_cells_csv, load_cells_geojson, read_wsi_offsets

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
        # GeoJSON coords are QuPath image-relative; offset applied inside
        cells["cell_x"] = cells["cell_x"] + x_off
        cells["cell_y"] = cells["cell_y"] + y_off
    else:
        cells = load_cells_csv(cells_path)
        # cells CSV coordinates are QuPath slide-relative (no offset needed)

    print(f"  cells loaded: {len(cells)}")

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

    radius_px = cfg.um_to_px(dilation_radius_um, pixel_size_microns)
    grid["is_peritumor_patch"] = mark_peritumor_patches(grid, radius_px)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(output_path, index=False)

    print(f"  tumor patches:    {int(grid['is_tumor'].sum())}")
    print(f"  peritumor patches:{int(grid['is_peritumor_patch'].sum())}")
    print(f"  immune cells assigned: {int(grid['cda_immune_count'].sum())}")
    print(f"  tumor cells assigned:  {int(grid['cda_tumor_cell_count'].sum())}")
    print(f"  wrote: {output_path}")
    print(f"=== done: {slide_id} ===")
    return grid
