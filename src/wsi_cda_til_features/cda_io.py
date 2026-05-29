"""Read CDA cell-detection outputs: compact cells CSV and GeoJSON fallback."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Cells CSV (primary format written by the Groovy script)
# ---------------------------------------------------------------------------

_CSV_COLS = ["cell_x", "cell_y", "cell_class"]
_TRUSTED_CLASSES = {"immune", "tumor_cell"}


def load_cells_csv(csv_path: Path) -> pd.DataFrame:
    """Read compact cells CSV. Only immune and tumor_cell rows are returned."""
    df = pd.read_csv(csv_path, usecols=_CSV_COLS, dtype={"cell_class": str})
    df = df[df["cell_class"].isin(_TRUSTED_CLASSES)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# GeoJSON fallback (when cells CSV is absent)
# ---------------------------------------------------------------------------


def _get_feature_label(properties: dict) -> str:
    if not properties:
        return ""
    cls = properties.get("classification")
    if isinstance(cls, dict):
        name = cls.get("name")
        if name:
            return str(name)
    for key in ("name", "class", "class_name", "classification",
                "pathClass", "objectType", "label"):
        val = properties.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, dict) and val.get("name"):
            return str(val["name"])
    return ""


def _normalize_label(label: str) -> str:
    s = (
        str(label).strip().lower()
        .replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
    )
    if any(t in s for t in ("immune", "til", "lymphocyte", "lymph")):
        return "immune"
    if any(t in s for t in ("tumor", "tumour", "melanoma", "malignant")):
        return "tumor_cell"
    if any(t in s for t in ("stroma", "stromal", "fibroblast")):
        return "stroma"
    if "other" in s:
        return "other"
    return "unknown"


def _iter_polygons(geom):
    from shapely.geometry import shape as _shape  # local import; shapely optional
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            if not part.is_empty:
                yield part
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_polygons(part)


def load_cells_geojson(
    geojson_path: Path,
    x_offset: int = 0,
    y_offset: int = 0,
) -> pd.DataFrame:
    """Read QuPath CDA detection GeoJSON; return centroid coordinates.

    x_offset / y_offset: OpenSlide bounds offsets to convert QuPath
    image-relative coordinates to slide-absolute coordinates.
    """
    from shapely.geometry import shape as _shape

    if not geojson_path.exists():
        raise FileNotFoundError(f"CDA GeoJSON not found: {geojson_path}")

    with open(geojson_path, encoding="utf-8") as fh:
        data = json.load(fh)

    rows: list[dict] = []
    raw_counts: Counter = Counter()
    norm_counts: Counter = Counter()

    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        raw_label = _get_feature_label(props)
        norm_label = _normalize_label(raw_label)
        raw_counts[raw_label or "<missing>"] += 1
        geom = _shape(feat["geometry"])
        for poly in _iter_polygons(geom):
            c = poly.centroid
            rows.append({
                "cell_x": float(c.x + x_offset),
                "cell_y": float(c.y + y_offset),
                "raw_label": raw_label,
                "cell_class": norm_label,
            })
            norm_counts[norm_label] += 1

    if not rows:
        raise ValueError(f"No CDA cell polygons found in {geojson_path}")

    df = pd.DataFrame(rows)
    df = df[df["cell_class"].isin(_TRUSTED_CLASSES)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# OpenSlide offset helper (optional; only needed when using GeoJSON fallback)
# ---------------------------------------------------------------------------


def read_wsi_offsets(wsi_path: Path) -> tuple[int, int]:
    """Return (x_off, y_off) bounds offsets from an OpenSlide-readable WSI."""
    try:
        import openslide
    except ImportError as exc:
        raise ImportError(
            "openslide-python is required to read WSI bounds offsets. "
            "Install it with: pip install openslide-python"
        ) from exc

    slide = openslide.OpenSlide(str(wsi_path))
    x_off = int(slide.properties.get("openslide.bounds-x", 0))
    y_off = int(slide.properties.get("openslide.bounds-y", 0))
    slide.close()
    return x_off, y_off
