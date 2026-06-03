"""Load and inspect tumor ROI GeoJSON masks produced by wsi-prototype-tumor-masker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_geojson(path: Path) -> dict:
    """Load a GeoJSON file and return the parsed dict."""
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def geojson_feature_count(data: dict) -> int:
    """Return the number of features in a GeoJSON FeatureCollection."""
    if data.get("type") == "FeatureCollection":
        return len(data.get("features", []))
    if data.get("type") == "Feature":
        return 1
    return 0


def geojson_classes(data: dict) -> list[str]:
    """Extract unique classification names from QuPath-exported GeoJSON."""
    names: list[str] = []
    features = (
        data.get("features", [])
        if data.get("type") == "FeatureCollection"
        else [data]
    )
    for feat in features:
        props = (feat.get("properties") or {})
        cls = props.get("classification")
        if isinstance(cls, dict):
            name = cls.get("name")
            if name and name not in names:
                names.append(str(name))
        elif isinstance(cls, str) and cls and cls not in names:
            names.append(cls)
    return names


def validate_roi_geojson(path: Path) -> None:
    """Raise ValueError if the file is not a valid non-empty GeoJSON."""
    try:
        data = load_geojson(path)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read ROI GeoJSON {path}: {exc}") from exc
    if geojson_feature_count(data) == 0:
        raise ValueError(f"ROI GeoJSON has no features: {path}")


def load_shapely_geometry(path: Path):
    """Load a GeoJSON file and return a unified Shapely geometry, or None if absent/empty."""
    if path is None or not path.exists():
        return None
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError:
        return None
    data = load_geojson(path)
    features = (
        data.get("features", [])
        if data.get("type") == "FeatureCollection"
        else [data]
    )
    geoms = [shape(f["geometry"]) for f in features if f.get("geometry")]
    if not geoms:
        return None
    merged = unary_union(geoms)
    return None if merged.is_empty else merged


def load_core_geojson(path: Path):
    """Load core.geojson and return a unified Shapely geometry, or None.

    The core GeoJSON is the selected tumor ROI.  It is used for intratumoral
    (TIL) analysis and for computing the area of the tumor core.
    """
    return load_shapely_geometry(path)


def load_peritumor_geojson(path: Path):
    """Load peritumor.geojson and return a unified Shapely geometry, or None.

    The peritumor GeoJSON is the CDA input ROI.  It covers the tumor plus the
    surrounding peritumoral band (default 300 µm).  The ring used for
    peritumoral immune-cell counts is computed as ``peritumor − core``.
    """
    return load_shapely_geometry(path)
