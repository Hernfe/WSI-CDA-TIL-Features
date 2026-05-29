"""Convenience wrapper: run-cda → overlay → extract-features in sequence."""

from __future__ import annotations

import sys
from pathlib import Path

from . import config as cfg
from .manifest import load_manifest, discover_slides, iter_rows, resolve_path


def run_pipeline(
    *,
    # run-cda inputs
    qupath_bin: str,
    groovy_script: Path,
    slides_dir: Path | None = None,
    roi_masks_dir: Path | None = None,
    tumor_grids_dir: Path | None = None,
    output_dir: Path,
    manifest_csv: Path | None = None,
    slide_ext: str = ".mrxs",
    roi_pattern: str = "{slide_id}_selected_cdaroi_stage5b.geojson",
    tumor_grid_pattern: str = "{slide_id}_selected_tumorgrid_stage5b.parquet",
    cells_pattern: str = "{slide_id}_cells.csv",
    overlay_pattern: str = "{slide_id}_overlay.parquet",
    component_pattern: str = "{slide_id}_component_selection_stage5b.json",
    component_json_dir: Path | None = None,
    measurements_csv_path: Path | None = None,
    export_geojson: bool = False,
    force: bool = False,
    dry_run: bool = False,
    patch_size: int = cfg.PATCH_SIZE,
    pixel_size_microns: float = cfg.PIXEL_SIZE_MICRONS,
    object_classifier_path: Path | None = None,
    pixel_classifier_path: Path | None = None,
) -> int:
    """Execute all three stages sequentially. Returns 0 on success, 1 on failures."""
    from .qupath_runner import run_one as run_cda_one
    from .overlay import run_overlay
    from .features import extract_features_batch

    cda_dir = output_dir / "cda"
    overlay_dir = output_dir / "overlays"
    features_dir = output_dir / "slide_features"
    measurements_csv = measurements_csv_path or (cda_dir / "measurements.csv")

    # --- Resolve slide list ---
    if manifest_csv is not None:
        manifest = load_manifest(manifest_csv)
    elif slides_dir is not None and roi_masks_dir is not None:
        manifest = discover_slides(slides_dir, roi_masks_dir, roi_pattern, slide_ext)
    else:
        print("ERROR: provide --manifest or both --slides and --roi-masks", file=sys.stderr)
        return 1

    slide_ids = list(manifest["slide_id"])
    if not slide_ids:
        print("ERROR: no slides found", file=sys.stderr)
        return 1

    print(f"Pipeline: {len(slide_ids)} slides")

    # --- Stage 1: run-cda ---
    cda_failures: list[str] = []
    for row in iter_rows(manifest):
        sid = row["slide_id"]
        wsi_path = Path(row["wsi_path"]) if row.get("wsi_path") else None
        roi_geojson = (
            Path(row["roi_geojson"])
            if row.get("roi_geojson")
            else resolve_path(roi_masks_dir or Path("."), roi_pattern, sid)
        )
        cells_csv = resolve_path(cda_dir, cells_pattern, sid)
        out_geojson = resolve_path(cda_dir, "{slide_id}_detections.geojson", sid)

        if wsi_path is None:
            print(f"  [{sid}] no wsi_path; skipping run-cda")
            cda_failures.append(sid)
            continue

        ok = run_cda_one(
            slide_id=sid,
            wsi_path=wsi_path,
            roi_geojson=roi_geojson,
            cells_csv=cells_csv,
            measurements_csv=measurements_csv,
            groovy_script=groovy_script,
            qupath_bin=qupath_bin,
            export_geojson=export_geojson,
            out_geojson=out_geojson if export_geojson else None,
            object_classifier_path=object_classifier_path,
            pixel_classifier_path=pixel_classifier_path,
            force=force,
            dry_run=dry_run,
        )
        if not ok:
            cda_failures.append(sid)

    # --- Stage 2: overlay ---
    overlay_failures: list[str] = []
    for row in iter_rows(manifest):
        sid = row["slide_id"]
        cells_csv = resolve_path(cda_dir, cells_pattern, sid)
        grid_path = (
            Path(row["tumor_grid"])
            if row.get("tumor_grid")
            else (
                resolve_path(tumor_grids_dir, tumor_grid_pattern, sid)
                if tumor_grids_dir else None
            )
        )
        if grid_path is None:
            print(f"  [{sid}] no tumor_grid path; skipping overlay", file=sys.stderr)
            overlay_failures.append(sid)
            continue

        try:
            run_overlay(
                slide_id=sid,
                tumor_grid_path=grid_path,
                cells_path=cells_csv,
                output_path=resolve_path(overlay_dir, overlay_pattern, sid),
                patch_size=patch_size,
                pixel_size_microns=pixel_size_microns,
            )
        except Exception as exc:
            print(f"  WARNING [{sid}] overlay failed: {exc}", file=sys.stderr)
            overlay_failures.append(sid)

    # --- Stage 3: extract-features ---
    try:
        extract_features_batch(
            slide_ids=slide_ids,
            overlay_dir=overlay_dir,
            output_path=features_dir / "final_features.csv",
            overlay_pattern=overlay_pattern,
            component_json_dir=component_json_dir,
            component_pattern=component_pattern,
            measurements_csv=measurements_csv if measurements_csv.exists() else None,
            patch_size=patch_size,
            pixel_size_microns=pixel_size_microns,
        )
    except Exception as exc:
        print(f"ERROR in extract-features: {exc}", file=sys.stderr)
        return 1

    total_failures = len(set(cda_failures + overlay_failures))
    if total_failures:
        print(
            f"\nPipeline done with {total_failures} slide failures. "
            "Check stderr for details."
        )
        return 1
    print("\nPipeline done: all slides succeeded.")
    return 0
