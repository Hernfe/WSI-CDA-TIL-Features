"""Extract slide-level engineered TIL / spatial features from overlay parquets.

Terminology:
  TIL (tumor-infiltrating leukocyte) — leukocytes inside the core tumor mask only.
  Peritumoral leukocytes / peritumoral immune cells — immune cells in the
      peritumoral ring (peritumor ROI minus core).  These are NOT called TILs.

Intratumoral features use the ``is_tumor`` patches from the overlay parquet.
Peritumoral counts come from the region summary JSON written by the overlay step:
  ring_immune_count = total_immune_in_CDA_roi - core_immune_count
  ring_area_mm2     = peritumor_area - core_area
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import label as scipy_label
from sklearn.cluster import DBSCAN

from . import config as cfg


# ---------------------------------------------------------------------------
# Section detection (DBSCAN on tumor-patch centroids)
# ---------------------------------------------------------------------------


def detect_sections(
    tumor_df: pd.DataFrame,
    patch_size: int = cfg.PATCH_SIZE,
    eps_um: float = cfg.SECTION_DBSCAN_EPS_UM,
    min_samples: int = cfg.SECTION_DBSCAN_MIN_SAMPLES,
    pixel_size: float = cfg.PIXEL_SIZE_MICRONS,
) -> tuple[pd.Series, int]:
    """DBSCAN on tumor patch centroids. Noise points snap to nearest cluster."""
    if tumor_df.empty:
        return pd.Series([], dtype=int), 0

    eps_px = cfg.um_to_px(eps_um, pixel_size)
    centers = np.column_stack([
        tumor_df["x"].to_numpy(dtype=float) + patch_size / 2,
        tumor_df["y"].to_numpy(dtype=float) + patch_size / 2,
    ])
    db = DBSCAN(eps=eps_px, min_samples=min_samples).fit(centers)
    labels = db.labels_.copy()

    real_labels = sorted({l for l in labels if l != -1})
    if not real_labels:
        return pd.Series(np.zeros(len(tumor_df), dtype=int), index=tumor_df.index), 1

    section_centroids = np.array(
        [centers[labels == l].mean(axis=0) for l in real_labels]
    )
    noise_mask = labels == -1
    if noise_mask.any():
        noise_pts = centers[noise_mask]
        diffs = noise_pts[:, None, :] - section_centroids[None, :, :]
        d = np.sqrt((diffs ** 2).sum(axis=-1))
        nearest = d.argmin(axis=1)
        labels[noise_mask] = np.array(real_labels)[nearest]

    return pd.Series(labels, index=tumor_df.index), len(real_labels)


def _primary_section(tumor_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    if tumor_df.empty:
        return tumor_df
    labels, _ = detect_sections(tumor_df, **kwargs)
    df = tumor_df.copy()
    df["_sec"] = labels.values
    primary = df["_sec"].value_counts().idxmax()
    return df[df["_sec"] == primary].drop(columns=["_sec"])


# ---------------------------------------------------------------------------
# Connected-component analysis within a section
# ---------------------------------------------------------------------------


def components_within_section(
    section_df: pd.DataFrame,
    patch_area: float,
) -> dict[str, Any]:
    """8-connected component analysis using compact grid adjacency."""
    xs = section_df["x"].to_numpy(dtype=np.int64)
    ys = section_df["y"].to_numpy(dtype=np.int64)

    unique_x = np.sort(np.unique(xs))
    unique_y = np.sort(np.unique(ys))
    cols = np.searchsorted(unique_x, xs)
    rows = np.searchsorted(unique_y, ys)

    grid = np.zeros((len(unique_y), len(unique_x)), dtype=np.uint8)
    grid[rows, cols] = 1

    labeled, n_cc = scipy_label(grid, structure=np.ones((3, 3), dtype=int))
    patch_labels = labeled[rows, cols]

    comps: list[tuple[np.ndarray, np.ndarray]] = []
    for cc_id in range(1, n_cc + 1):
        mask = patch_labels == cc_id
        comps.append((xs[mask], ys[mask]))
    comps.sort(key=lambda c: len(c[0]), reverse=True)

    main_xs, main_ys = comps[0]
    n_main = len(main_xs)
    main_centroid = np.array([main_xs.mean(), main_ys.mean()])
    centroids = [np.array([c[0].mean(), c[1].mean()]) for c in comps]
    dists = [float(np.sqrt(((c - main_centroid) ** 2).sum())) for c in centroids]
    satellite_patches = sum(len(c[0]) for c in comps[1:])

    return {
        "n_components":           n_cc,
        "n_satellites":           max(0, n_cc - 1),
        "main_component_patches": n_main,
        "main_component_area_mm2":  n_main * patch_area,
        "satellite_area_mm2":       satellite_patches * patch_area,
        "satellite_area_fraction":  satellite_patches / max(1, len(section_df)),
        "mean_satellite_distance_px": float(np.mean(dists[1:])) if len(dists) > 1 else 0.0,
        "max_satellite_distance_px":  float(np.max(dists[1:])) if len(dists) > 1 else 0.0,
        "centroid":  main_centroid,
        "n_patches": len(section_df),
    }


# ---------------------------------------------------------------------------
# Section-level features
# ---------------------------------------------------------------------------


def section_features(
    tumor_df: pd.DataFrame,
    patch_area: float | None = None,
    **detect_kwargs,
) -> dict[str, Any]:
    if patch_area is None:
        patch_area = cfg.patch_area_mm2()

    null_out: dict[str, Any] = {
        "n_sections": 0,
        "multi_section_flag": False,
        "max_inter_section_distance_px": 0.0,
        "secondary_sections_area_mm2": 0.0,
        "primary_n_components": 0,
        "primary_section_n_cc_satellites": 0,
        "primary_main_component_patches": 0,
        "primary_main_component_area_mm2": 0.0,
        "primary_satellite_area_mm2": 0.0,
        "primary_satellite_area_fraction": 0.0,
        "primary_mean_satellite_distance_px": 0.0,
        "primary_max_satellite_distance_px": 0.0,
    }
    if tumor_df.empty:
        return null_out

    labels, n_sections = detect_sections(tumor_df, **detect_kwargs)
    df = tumor_df.copy()
    df["_sec"] = labels.values

    sections = sorted(
        [
            components_within_section(df[df["_sec"] == lab], patch_area)
            for lab in df["_sec"].unique()
        ],
        key=lambda s: s["n_patches"],
        reverse=True,
    )
    primary = sections[0]

    max_inter = 0.0
    if n_sections > 1:
        centroids = np.array([s["centroid"] for s in sections])
        diffs = centroids[:, None, :] - centroids[None, :, :]
        max_inter = float(np.sqrt((diffs ** 2).sum(axis=-1)).max())

    return {
        "n_sections":               n_sections,
        "multi_section_flag":       n_sections > 1,
        "max_inter_section_distance_px": max_inter,
        "secondary_sections_area_mm2": float(
            sum(s["n_patches"] for s in sections[1:]) * patch_area
        ),
        "primary_n_components":               primary["n_components"],
        "primary_section_n_cc_satellites":    primary["n_satellites"],
        "primary_main_component_patches":     primary["main_component_patches"],
        "primary_main_component_area_mm2":    float(primary["main_component_area_mm2"]),
        "primary_satellite_area_mm2":         float(primary["satellite_area_mm2"]),
        "primary_satellite_area_fraction":    float(primary["satellite_area_fraction"]),
        "primary_mean_satellite_distance_px": float(primary["mean_satellite_distance_px"]),
        "primary_max_satellite_distance_px":  float(primary["max_satellite_distance_px"]),
    }


# ---------------------------------------------------------------------------
# TIL spatial dispersion within primary section
# ---------------------------------------------------------------------------


def primary_section_til_dispersion(
    tumor_df: pd.DataFrame,
    patch_size: int = cfg.PATCH_SIZE,
    **detect_kwargs,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "primary_section_til_dispersion_mean_distance_px": np.nan,
        "primary_section_til_dispersion_max_distance_px": np.nan,
    }
    if tumor_df.empty:
        return out

    primary = _primary_section(tumor_df, patch_size=patch_size, **detect_kwargs)
    weights = primary["cda_immune_count"].to_numpy(dtype=float)
    keep = weights > 0
    if keep.sum() == 0:
        return out

    half = patch_size / 2
    x = primary.loc[keep, "x"].to_numpy(dtype=float) + half
    y = primary.loc[keep, "y"].to_numpy(dtype=float) + half
    w = weights[keep]
    cx = np.average(x, weights=w)
    cy = np.average(y, weights=w)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    out["primary_section_til_dispersion_mean_distance_px"] = float(np.average(d, weights=w))
    out["primary_section_til_dispersion_max_distance_px"] = float(d.max())
    return out


# ---------------------------------------------------------------------------
# Component-selection metadata (stage5b JSON)
# ---------------------------------------------------------------------------


def load_component_selection_metadata(json_path: Path) -> dict[str, Any]:
    """Read component_selection_stage5b.json. Returns {} if not found."""
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as fh:
            d = json.load(fh)
        return {
            "sel_uncertainty":          int(bool(d.get("uncertainty", False))),
            "sel_n_satellite_ccs":      len(d.get("selected_satellite_cc_ids", [])),
            "sel_group_median_margin":  d.get("group_median_margin"),
            "sel_group_high_conf_frac": d.get("group_high_conf_frac"),
            "sel_group_compactness":    d.get("group_compactness"),
            "sel_stage5_area_mm2":      d.get("total_area_mm2"),
        }
    except Exception as exc:
        print(f"  warning: could not read component_selection JSON {json_path}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Region summary (written by overlay step)
# ---------------------------------------------------------------------------


def load_region_summary(json_path: Path) -> dict[str, Any]:
    """Read {slide_id}_region_summary.json. Returns {} if not found.

    The region summary contains:
      core_immune_count, core_tumor_cell_count  — cells assigned to tumor patches
      ring_immune_count, ring_tumor_cell_count  — total minus core
      core_area_mm2, peritumor_area_mm2, ring_area_mm2
    """
    if json_path is None or not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"  warning: could not read region summary {json_path}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# CDA summary (slide-level measurements.csv)
# ---------------------------------------------------------------------------


def load_cda_summary(measurements_csv: Path) -> pd.DataFrame:
    """Load QuPath image-level measurements CSV.

    Expects a column named 'Image' (or 'image', 'Slide', 'slide') containing
    the slide identifier.  Stroma and Other columns are dropped (lower-trust).
    """
    if not measurements_csv.exists():
        return pd.DataFrame(columns=["slide_id"])

    df = pd.read_csv(measurements_csv)
    image_col = next(
        (c for c in ("Image", "image", "Slide", "slide") if c in df.columns), None
    )
    if image_col is None:
        return pd.DataFrame(columns=["slide_id"])

    import re
    df = df.rename(columns={image_col: "slide_id"})
    df.columns = [
        "slide_id" if c == "slide_id"
        else "cda_summary_" + re.sub(r"[^a-zA-Z0-9]+", "_", str(c)).strip("_").lower()
        for c in df.columns
    ]
    drop_cols = [
        c for c in df.columns
        if c != "slide_id" and any(p in c for p in ("stroma", "other"))
    ]
    return df.drop(columns=drop_cols, errors="ignore")


# ---------------------------------------------------------------------------
# Per-slide feature extraction
# ---------------------------------------------------------------------------


def extract_slide_features(
    slide_id: str,
    overlay_path: Path,
    component_json_path: Path | None = None,
    region_summary_path: Path | None = None,
    wsi_filename: str | None = None,
    patch_size: int = cfg.PATCH_SIZE,
    pixel_size_microns: float = cfg.PIXEL_SIZE_MICRONS,
    til_thresholds: list[int] | None = None,
) -> dict[str, Any] | None:
    """Extract all features for one slide. Returns None if overlay missing."""
    if not overlay_path.exists():
        print(f"  [{slide_id}] overlay not found: {overlay_path}; skipping")
        return None

    if til_thresholds is None:
        til_thresholds = cfg.TIL_DENSITY_THRESHOLDS

    overlay = pd.read_parquet(overlay_path)
    overlay["x"] = overlay["x"].astype(int)
    overlay["y"] = overlay["y"].astype(int)

    tumor = overlay[overlay["is_tumor"]].copy()

    patch_area = ((patch_size * pixel_size_microns) / 1000.0) ** 2

    row: dict[str, Any] = {
        "slide_id":              slide_id,
        "wsi_filename":          wsi_filename or "",
        "patch_count_total":     int(len(overlay)),
        "patch_count_tumor":     int(len(tumor)),
        "tumor_area_mm2":        float(len(tumor) * patch_area),
        "total_cda_immune_cells": int(overlay["cda_immune_count"].sum()),
        "total_cda_tumor_cells":  int(overlay["cda_tumor_cell_count"].sum()),
    }

    if tumor.empty:
        row.update({
            "til_count":                             0,
            "til_density_per_mm2":                   np.nan,
            "intratumoral_leukocyte_count":          0,
            "intratumoral_leukocyte_density_per_mm2": np.nan,
            "mean_intratumoral_til_count_per_patch":  np.nan,
            "max_intratumoral_til_count_per_patch":   np.nan,
            "mean_intratumoral_til_density_per_mm2":  np.nan,
            "max_intratumoral_til_density_per_mm2":   np.nan,
            "mean_tumor_cell_density_per_mm2":        np.nan,
        })
        for thr in til_thresholds:
            row[f"fraction_tumor_patches_til_count_ge_{thr}"] = np.nan
    else:
        core_area_mm2 = float(len(tumor) * patch_area)
        til_count = int(tumor["cda_immune_count"].sum())
        row.update({
            "til_count":                             til_count,
            "til_density_per_mm2":                   til_count / core_area_mm2 if core_area_mm2 > 0 else np.nan,
            "intratumoral_leukocyte_count":          til_count,
            "intratumoral_leukocyte_density_per_mm2": til_count / core_area_mm2 if core_area_mm2 > 0 else np.nan,
            "mean_intratumoral_til_count_per_patch":
                float(tumor["cda_immune_count"].mean()),
            "max_intratumoral_til_count_per_patch":
                float(tumor["cda_immune_count"].max()),
            "mean_intratumoral_til_density_per_mm2":
                float(tumor["cda_immune_density_per_mm2"].mean()),
            "max_intratumoral_til_density_per_mm2":
                float(tumor["cda_immune_density_per_mm2"].max()),
            "mean_tumor_cell_density_per_mm2":
                float(tumor["cda_tumor_cell_density_per_mm2"].mean()),
        })
        for thr in til_thresholds:
            row[f"fraction_tumor_patches_til_count_ge_{thr}"] = float(
                (tumor["cda_immune_count"] >= thr).mean()
            )

    # Peritumoral features from region summary (ring = peritumor ROI minus core).
    rs = load_region_summary(region_summary_path) if region_summary_path else {}
    ring_area_mm2 = float(rs.get("ring_area_mm2") or 0.0)
    ring_immune = rs.get("ring_immune_count")
    ring_tumor_cell = rs.get("ring_tumor_cell_count")

    if ring_area_mm2 > 0 and ring_immune is not None:
        peri_leuk_count = int(ring_immune)
        peri_leuk_density = peri_leuk_count / ring_area_mm2
    else:
        peri_leuk_count = int(ring_immune) if ring_immune is not None else None
        peri_leuk_density = np.nan

    row.update({
        "peritumoral_leukocyte_count":
            peri_leuk_count if peri_leuk_count is not None else np.nan,
        "peritumoral_leukocyte_density_per_mm2": peri_leuk_density,
        "peritumoral_immune_cell_count":
            peri_leuk_count if peri_leuk_count is not None else np.nan,
        "peritumoral_immune_cell_density_per_mm2": peri_leuk_density,
        "peritumoral_ring_area_mm2": ring_area_mm2 if ring_area_mm2 > 0 else np.nan,
        "peritumoral_ring_patch_count":
            round(ring_area_mm2 / patch_area) if ring_area_mm2 > 0 else np.nan,
    })

    detect_kwargs = dict(patch_size=patch_size, pixel_size=pixel_size_microns)
    row.update(section_features(tumor, patch_area=patch_area, **detect_kwargs))
    row.update(primary_section_til_dispersion(tumor, **detect_kwargs))

    if component_json_path is not None:
        row.update(load_component_selection_metadata(component_json_path))

    return row


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def extract_features_batch(
    slide_ids: list[str],
    overlay_dir: Path,
    output_path: Path,
    overlay_pattern: str = "{slide_id}_overlay.parquet",
    component_json_dir: Path | None = None,
    component_pattern: str = "{slide_id}_component_selection_stage5b.json",
    region_summary_dir: Path | None = None,
    region_summary_pattern: str = "{slide_id}_region_summary.json",
    measurements_csv: Path | None = None,
    wsi_filenames: dict[str, str] | None = None,
    patch_size: int = cfg.PATCH_SIZE,
    pixel_size_microns: float = cfg.PIXEL_SIZE_MICRONS,
) -> pd.DataFrame:
    from .manifest import resolve_path

    cda_summary = (
        load_cda_summary(measurements_csv)
        if measurements_csv is not None
        else pd.DataFrame(columns=["slide_id"])
    )

    # Default region_summary_dir to overlay_dir (overlay writes summary alongside parquet).
    _summary_dir = region_summary_dir if region_summary_dir is not None else overlay_dir

    rows: list[dict] = []
    skipped: list[str] = []

    for slide_id in slide_ids:
        overlay_path = resolve_path(overlay_dir, overlay_pattern, slide_id)
        comp_path = (
            resolve_path(component_json_dir, component_pattern, slide_id)
            if component_json_dir is not None
            else None
        )
        summary_path = resolve_path(_summary_dir, region_summary_pattern, slide_id)
        wsi_fn = (wsi_filenames or {}).get(slide_id)
        try:
            r = extract_slide_features(
                slide_id=slide_id,
                overlay_path=overlay_path,
                component_json_path=comp_path,
                region_summary_path=summary_path,
                wsi_filename=wsi_fn,
                patch_size=patch_size,
                pixel_size_microns=pixel_size_microns,
            )
            if r is not None:
                rows.append(r)
        except Exception as exc:
            print(f"  WARNING [{slide_id}]: {exc}; skipping")
            skipped.append(slide_id)

    if not rows:
        raise RuntimeError(
            "No slides produced features. Check overlay parquets in: " + str(overlay_dir)
        )

    df = pd.DataFrame(rows)
    if not cda_summary.empty and len(cda_summary.columns) > 1:
        df = df.merge(cda_summary, on="slide_id", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  wrote: {output_path}  ({len(df)} rows, {len(df.columns)} cols)")
    if skipped:
        print(f"  skipped ({len(skipped)}): {skipped}")
    return df
