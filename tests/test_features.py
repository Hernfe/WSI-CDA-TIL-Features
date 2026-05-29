"""Tests for slide-level feature extraction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wsi_cda_til_features.features import (
    detect_sections,
    section_features,
    primary_section_til_dispersion,
    extract_slide_features,
    load_component_selection_metadata,
)


PATCH_SIZE = 224
PIXEL_SIZE = 0.22


def _make_tumor_df(
    n_x: int = 5,
    n_y: int = 4,
    immune_per_patch: int = 3,
    tumor_cell_per_patch: int = 5,
    patch_size: int = PATCH_SIZE,
) -> pd.DataFrame:
    rows = []
    for xi in range(n_x):
        for yi in range(n_y):
            rows.append({
                "x": xi * patch_size,
                "y": yi * patch_size,
                "cda_immune_count": immune_per_patch,
                "cda_tumor_cell_count": tumor_cell_per_patch,
            })
    return pd.DataFrame(rows)


def _make_overlay_parquet(tmp_path: Path, slide_id: str) -> Path:
    grid = _make_tumor_df()
    grid["is_tumor"] = True
    grid["is_peritumor_patch"] = False
    grid["predicted_class"] = "Tumor"
    grid["confidence"] = 0.9
    patch_area = ((PATCH_SIZE * PIXEL_SIZE) / 1000.0) ** 2
    grid["patch_area_mm2"] = patch_area
    grid["cda_total_cells"] = grid["cda_immune_count"] + grid["cda_tumor_cell_count"]
    grid["cda_immune_density_per_mm2"] = grid["cda_immune_count"] / patch_area
    grid["cda_tumor_cell_density_per_mm2"] = grid["cda_tumor_cell_count"] / patch_area
    path = tmp_path / f"{slide_id}_overlay.parquet"
    grid.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------


def test_detect_sections_single_cluster():
    tumor = _make_tumor_df()
    labels, n_sections = detect_sections(
        tumor, patch_size=PATCH_SIZE, pixel_size=PIXEL_SIZE
    )
    assert n_sections >= 1
    assert len(labels) == len(tumor)


def test_detect_sections_empty():
    labels, n_sections = detect_sections(pd.DataFrame())
    assert n_sections == 0
    assert len(labels) == 0


def test_section_features_basic():
    tumor = _make_tumor_df()
    patch_area = ((PATCH_SIZE * PIXEL_SIZE) / 1000.0) ** 2
    feats = section_features(tumor, patch_area=patch_area,
                              patch_size=PATCH_SIZE, pixel_size=PIXEL_SIZE)
    assert feats["n_sections"] >= 1
    assert isinstance(feats["multi_section_flag"], bool)
    assert feats["primary_n_components"] >= 1


def test_section_features_empty():
    feats = section_features(pd.DataFrame())
    assert feats["n_sections"] == 0
    assert feats["multi_section_flag"] is False


def test_primary_section_til_dispersion():
    tumor = _make_tumor_df()
    result = primary_section_til_dispersion(
        tumor, patch_size=PATCH_SIZE, pixel_size=PIXEL_SIZE
    )
    assert "primary_section_til_dispersion_mean_distance_px" in result
    assert "primary_section_til_dispersion_max_distance_px" in result


def test_primary_section_til_dispersion_no_immune():
    tumor = _make_tumor_df(immune_per_patch=0)
    result = primary_section_til_dispersion(tumor)
    assert np.isnan(result["primary_section_til_dispersion_mean_distance_px"])


# ---------------------------------------------------------------------------
# extract_slide_features
# ---------------------------------------------------------------------------


def test_extract_slide_features_basic(tmp_path):
    overlay_path = _make_overlay_parquet(tmp_path, "example_slide_001")
    row = extract_slide_features(
        slide_id="example_slide_001",
        overlay_path=overlay_path,
        patch_size=PATCH_SIZE,
        pixel_size_microns=PIXEL_SIZE,
    )
    assert row is not None
    assert row["slide_id"] == "example_slide_001"
    assert row["patch_count_tumor"] == 20
    assert row["total_cda_immune_cells"] == 60  # 20 patches × 3
    assert row["total_cda_tumor_cells"] == 100   # 20 patches × 5
    assert row["mean_intratumoral_til_count_per_patch"] == pytest.approx(3.0)
    assert row["fraction_tumor_patches_til_count_ge_5"] == pytest.approx(0.0)


def test_extract_slide_features_missing_overlay(tmp_path):
    result = extract_slide_features(
        "example_slide_001",
        tmp_path / "nonexistent_overlay.parquet",
    )
    assert result is None


def test_extract_slide_features_empty_tumor(tmp_path):
    grid = _make_tumor_df()
    grid["is_tumor"] = False
    grid["is_peritumor_patch"] = False
    grid["predicted_class"] = "Non-tumor"
    grid["confidence"] = 0.5
    patch_area = ((PATCH_SIZE * PIXEL_SIZE) / 1000.0) ** 2
    grid["patch_area_mm2"] = patch_area
    grid["cda_total_cells"] = grid["cda_immune_count"] + grid["cda_tumor_cell_count"]
    grid["cda_immune_density_per_mm2"] = grid["cda_immune_count"] / patch_area
    grid["cda_tumor_cell_density_per_mm2"] = grid["cda_tumor_cell_count"] / patch_area
    path = tmp_path / "example_slide_001_overlay.parquet"
    grid.to_parquet(path, index=False)
    row = extract_slide_features("example_slide_001", path,
                                  patch_size=PATCH_SIZE, pixel_size_microns=PIXEL_SIZE)
    assert row is not None
    assert row["patch_count_tumor"] == 0
    assert np.isnan(row["mean_intratumoral_til_count_per_patch"])


# ---------------------------------------------------------------------------
# Component selection metadata
# ---------------------------------------------------------------------------


def test_load_component_selection_metadata(tmp_path):
    data = {
        "uncertainty": False,
        "selected_satellite_cc_ids": [1, 2],
        "group_median_margin": 0.12,
        "group_high_conf_frac": 0.85,
        "group_compactness": 0.72,
        "total_area_mm2": 5.4,
    }
    p = tmp_path / "example_slide_001_component_selection_stage5b.json"
    p.write_text(json.dumps(data))
    meta = load_component_selection_metadata(p)
    assert meta["sel_n_satellite_ccs"] == 2
    assert meta["sel_stage5_area_mm2"] == pytest.approx(5.4)


def test_load_component_selection_metadata_missing(tmp_path):
    meta = load_component_selection_metadata(tmp_path / "missing.json")
    assert meta == {}


# ---------------------------------------------------------------------------
# QuPath command construction (dry-run)
# ---------------------------------------------------------------------------


def test_qupath_command_construction():
    from wsi_cda_til_features.qupath_runner import build_command
    cmd = build_command(
        slide_id="example_slide_001",
        wsi_path=Path("/slides/example_slide_001.mrxs"),
        roi_geojson=Path("/masks/example_slide_001_selected_cdaroi_stage5b.geojson"),
        cells_csv=Path("/out/cda/example_slide_001_cells.csv"),
        measurements_csv=Path("/out/cda/measurements.csv"),
        groovy_script=Path("scripts/run_cda_one_slide.groovy"),
        qupath_bin="/usr/bin/QuPath",
        export_geojson=False,
    )
    assert "example_slide_001" in cmd
    assert "roi_geojson=" in cmd
    assert "cells_csv=" in cmd
    assert "export_geojson=false" in cmd
    assert "private" not in cmd.lower()
    assert "/data/melanoma" not in cmd
