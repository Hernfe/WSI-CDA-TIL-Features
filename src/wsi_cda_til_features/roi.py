"""Load and inspect tumor ROI GeoJSON masks (stage5b output)."""

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
