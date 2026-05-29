"""Tests for manifest parsing and slide discovery."""

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from wsi_cda_til_features.manifest import (
    load_manifest,
    discover_slides,
    pattern_to_path,
    resolve_path,
)


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "manifest.csv"
    p.write_text(textwrap.dedent(content).strip())
    return p


def test_load_manifest_basic(tmp_path):
    csv = _write_csv(tmp_path, """
        slide_id,wsi_path,roi_geojson
        example_slide_001,/slides/s1.mrxs,/masks/s1.geojson
        example_slide_002,/slides/s2.mrxs,/masks/s2.geojson
    """)
    df = load_manifest(csv)
    assert list(df["slide_id"]) == ["example_slide_001", "example_slide_002"]
    assert "wsi_path" in df.columns


def test_load_manifest_missing_slide_id_column(tmp_path):
    csv = _write_csv(tmp_path, """
        wsi_path,roi_geojson
        /slides/s1.mrxs,/masks/s1.geojson
    """)
    with pytest.raises(ValueError, match="slide_id"):
        load_manifest(csv)


def test_load_manifest_blank_rows_stripped(tmp_path):
    csv = _write_csv(tmp_path, """
        slide_id,wsi_path
        example_slide_001,/slides/s1.mrxs
        ,
        example_slide_002,/slides/s2.mrxs
    """)
    df = load_manifest(csv)
    assert len(df) == 2


def test_pattern_to_path():
    p = pattern_to_path("{slide_id}_selected_cdaroi_stage5b.geojson", "example_slide_001")
    assert p == Path("example_slide_001_selected_cdaroi_stage5b.geojson")


def test_resolve_path_relative(tmp_path):
    p = resolve_path(tmp_path, "{slide_id}_cells.csv", "example_slide_001")
    assert p == tmp_path / "example_slide_001_cells.csv"


def test_discover_slides(tmp_path):
    slides_dir = tmp_path / "slides"
    masks_dir = tmp_path / "masks"
    slides_dir.mkdir()
    masks_dir.mkdir()

    (slides_dir / "example_slide_001.mrxs").write_text("fake")
    (slides_dir / "example_slide_002.mrxs").write_text("fake")
    (masks_dir / "example_slide_001_selected_cdaroi_stage5b.geojson").write_text("{}")
    # no mask for 002 — should be excluded

    df = discover_slides(slides_dir, masks_dir)
    assert len(df) == 1
    assert df.iloc[0]["slide_id"] == "example_slide_001"


def test_discover_slides_empty(tmp_path):
    slides_dir = tmp_path / "slides"
    masks_dir = tmp_path / "masks"
    slides_dir.mkdir()
    masks_dir.mkdir()
    df = discover_slides(slides_dir, masks_dir)
    assert df.empty
