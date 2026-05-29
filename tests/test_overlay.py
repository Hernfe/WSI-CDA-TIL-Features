"""Tests for cell-to-patch overlay logic."""

import numpy as np
import pandas as pd
import pytest

from wsi_cda_til_features.overlay import (
    assign_cells_to_patches,
    mark_peritumor_patches,
    load_tumor_grid,
)


def _make_grid(n_x=5, n_y=4, patch_size=224, is_tumor_rows=None) -> pd.DataFrame:
    """Create a synthetic tumor grid DataFrame."""
    rows = []
    for xi in range(n_x):
        for yi in range(n_y):
            rows.append({
                "x": xi * patch_size,
                "y": yi * patch_size,
                "predicted_class": "Tumor",
                "confidence": 0.9,
                "is_tumor": True,
            })
    df = pd.DataFrame(rows)
    if is_tumor_rows is not None:
        df["is_tumor"] = False
        for i in is_tumor_rows:
            df.loc[i, "is_tumor"] = True
    return df


def _make_cells(xs, ys, classes) -> pd.DataFrame:
    return pd.DataFrame({"cell_x": xs, "cell_y": ys, "cell_class": classes})


def test_assign_cells_basic():
    grid = _make_grid()
    cells = _make_cells(
        xs=[10, 230, 460],
        ys=[10, 10, 10],
        classes=["immune", "tumor_cell", "immune"],
    )
    assigned = assign_cells_to_patches(cells, grid, patch_size=224)
    assert len(assigned) == 3
    assert set(assigned["cell_class"].unique()) == {"immune", "tumor_cell"}


def test_assign_cells_outside_grid():
    grid = _make_grid(n_x=2, n_y=2)
    cells = _make_cells(
        xs=[9999],
        ys=[9999],
        classes=["immune"],
    )
    assigned = assign_cells_to_patches(cells, grid, patch_size=224)
    assert assigned.empty


def test_assign_cells_empty_cells():
    grid = _make_grid()
    cells = pd.DataFrame(columns=["cell_x", "cell_y", "cell_class"])
    assigned = assign_cells_to_patches(cells, grid, patch_size=224)
    assert assigned.empty


def test_mark_peritumor_patches():
    pytest.importorskip("sklearn")
    grid = _make_grid(n_x=5, n_y=1, is_tumor_rows=[2])
    is_peri = mark_peritumor_patches(grid, radius_px=300)
    # Patches adjacent to is_tumor patch should be peritumor
    assert is_peri.any()
    # Tumor patch itself should not be marked as peritumor
    assert not is_peri[grid["is_tumor"]].any()


def test_mark_peritumor_no_tumor_patches():
    grid = _make_grid(n_x=3, n_y=1)
    grid["is_tumor"] = False
    is_peri = mark_peritumor_patches(grid, radius_px=300)
    assert not is_peri.any()


def test_load_tumor_grid_csv(tmp_path):
    csv = tmp_path / "grid.csv"
    csv.write_text(
        "x,y,predicted_class,confidence,is_tumor\n"
        "0,0,Tumor,0.9,True\n"
        "224,0,Tumor,0.85,True\n"
    )
    grid = load_tumor_grid(csv)
    assert len(grid) == 2
    assert "is_tumor" in grid.columns


def test_load_tumor_grid_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("x,y\n0,0\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_tumor_grid(csv)
