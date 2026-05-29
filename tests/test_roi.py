"""Tests for ROI GeoJSON loading."""

import json
from pathlib import Path

import pytest

from wsi_cda_til_features.roi import (
    load_geojson,
    geojson_feature_count,
    geojson_classes,
    validate_roi_geojson,
)


def _write_geojson(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "roi.geojson"
    p.write_text(json.dumps(data))
    return p


def test_load_geojson_basic(tmp_path):
    data = {"type": "FeatureCollection", "features": []}
    p = _write_geojson(tmp_path, data)
    result = load_geojson(p)
    assert result["type"] == "FeatureCollection"


def test_load_geojson_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_geojson(tmp_path / "nope.geojson")


def test_feature_count_collection(tmp_path):
    data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": None},
            {"type": "Feature", "properties": {}, "geometry": None},
        ],
    }
    assert geojson_feature_count(data) == 2


def test_feature_count_single_feature():
    assert geojson_feature_count({"type": "Feature"}) == 1


def test_geojson_classes_dict_classification():
    data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "properties": {"classification": {"name": "Tumor"}},
             "geometry": None},
            {"type": "Feature",
             "properties": {"classification": {"name": "Ignore*"}},
             "geometry": None},
        ],
    }
    classes = geojson_classes(data)
    assert "Tumor" in classes
    assert "Ignore*" in classes


def test_validate_roi_geojson_empty_collection(tmp_path):
    data = {"type": "FeatureCollection", "features": []}
    p = _write_geojson(tmp_path, data)
    with pytest.raises(ValueError, match="no features"):
        validate_roi_geojson(p)


def test_validate_roi_geojson_valid(tmp_path):
    data = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": None}],
    }
    p = _write_geojson(tmp_path, data)
    validate_roi_geojson(p)  # should not raise
