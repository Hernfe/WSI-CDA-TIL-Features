"""Tests for CDA cell CSV and GeoJSON loading."""

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from wsi_cda_til_features.cda_io import load_cells_csv, load_cells_geojson


def _cells_csv(tmp_path: Path) -> Path:
    p = tmp_path / "example_slide_001_cells.csv"
    p.write_text(textwrap.dedent("""
        slide_id,cell_x,cell_y,raw_class,cell_class,cell_area_px,cell_area_um2
        example_slide_001,100,200,Immune cells,immune,80,3.87
        example_slide_001,150,250,Tumor,tumor_cell,120,5.80
        example_slide_001,300,400,Stroma,stroma,90,4.35
        example_slide_001,500,600,Other,other,70,3.39
    """).strip())
    return p


def test_load_cells_csv_filters_trusted_classes(tmp_path):
    p = _cells_csv(tmp_path)
    df = load_cells_csv(p)
    assert set(df["cell_class"].unique()) == {"immune", "tumor_cell"}
    assert len(df) == 2


def test_load_cells_csv_columns(tmp_path):
    p = _cells_csv(tmp_path)
    df = load_cells_csv(p)
    assert {"cell_x", "cell_y", "cell_class"}.issubset(df.columns)


def test_load_cells_csv_empty_after_filter(tmp_path):
    p = tmp_path / "empty_cells.csv"
    p.write_text("slide_id,cell_x,cell_y,raw_class,cell_class,cell_area_px,cell_area_um2\n"
                 "example_slide_001,100,200,Stroma,stroma,80,3.87\n")
    df = load_cells_csv(p)
    assert df.empty


def _cells_geojson(tmp_path: Path) -> Path:
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": "Immune cells"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"classification": {"name": "Tumor"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 20], [21, 20], [21, 21], [20, 21], [20, 20]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"classification": {"name": "Stroma"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[30, 30], [31, 30], [31, 31], [30, 31], [30, 30]]],
                },
            },
        ],
    }
    p = tmp_path / "detections.geojson"
    p.write_text(json.dumps(data))
    return p


def test_load_cells_geojson_filters_trusted(tmp_path):
    pytest.importorskip("shapely")
    p = _cells_geojson(tmp_path)
    df = load_cells_geojson(p)
    assert set(df["cell_class"].unique()) == {"immune", "tumor_cell"}
    assert len(df) == 2


def test_load_cells_geojson_applies_offset(tmp_path):
    pytest.importorskip("shapely")
    p = _cells_geojson(tmp_path)
    df_no_off = load_cells_geojson(p)
    df_off    = load_cells_geojson(p, x_offset=1000, y_offset=2000)
    assert (df_off["cell_x"] - df_no_off["cell_x"]).abs().gt(0).any()


def test_load_cells_geojson_missing_file(tmp_path):
    pytest.importorskip("shapely")
    with pytest.raises(FileNotFoundError):
        load_cells_geojson(tmp_path / "nonexistent.geojson")
