"""Command-line interface: wsi-cda-til {run-cda,overlay,extract-features,run}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config as cfg


# ---------------------------------------------------------------------------
# run-cda
# ---------------------------------------------------------------------------


def cmd_run_cda(args: argparse.Namespace) -> int:
    from .manifest import load_manifest, discover_slides, iter_rows, resolve_path
    from .qupath_runner import run_one

    cda_dir = Path(args.output) / "cda"
    measurements_csv = cda_dir / "measurements.csv"

    if args.manifest:
        manifest = load_manifest(Path(args.manifest))
    elif args.slides and args.roi_masks:
        manifest = discover_slides(
            Path(args.slides),
            Path(args.roi_masks),
            roi_pattern=args.roi_pattern,
            slide_ext=args.slide_ext,
        )
    else:
        print(
            "ERROR: provide --manifest or both --slides and --roi-masks",
            file=sys.stderr,
        )
        return 1

    if manifest.empty:
        print("ERROR: no slides found", file=sys.stderr)
        return 1

    print(f"run-cda: {len(manifest)} slides")
    if args.dry_run:
        print("DRY RUN\n")

    ok = fail = 0
    for row in iter_rows(manifest):
        sid = row["slide_id"]
        wsi_path_str = row.get("wsi_path", "")
        wsi_path = Path(wsi_path_str) if wsi_path_str else None

        roi_str = row.get("roi_geojson", "")
        roi_geojson = (
            Path(roi_str)
            if roi_str
            else resolve_path(Path(args.roi_masks or "."), args.roi_pattern, sid)
        )
        cells_csv = resolve_path(cda_dir, "{slide_id}_cells.csv", sid)
        out_geojson = resolve_path(cda_dir, "{slide_id}_detections.geojson", sid)

        if wsi_path is None and not args.dry_run:
            print(f"  [{sid}] no wsi_path; skipping", file=sys.stderr)
            fail += 1
            continue

        success = run_one(
            slide_id=sid,
            wsi_path=wsi_path or Path(f"<wsi_{sid}>"),
            roi_geojson=roi_geojson,
            cells_csv=cells_csv,
            measurements_csv=measurements_csv,
            groovy_script=Path(args.groovy_script),
            qupath_bin=args.qupath_bin,
            export_geojson=args.export_geojson,
            out_geojson=out_geojson if args.export_geojson else None,
            object_classifier_path=(
                Path(args.object_classifier) if args.object_classifier else None
            ),
            pixel_classifier_path=(
                Path(args.pixel_classifier) if args.pixel_classifier else None
            ),
            force=args.force,
            dry_run=args.dry_run,
        )
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nrun-cda: {ok} ok, {fail} failed/skipped")
    return 0 if fail == 0 else 1


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


def cmd_overlay(args: argparse.Namespace) -> int:
    from .manifest import load_manifest, iter_rows, resolve_path
    from .overlay import run_overlay

    overlay_dir = Path(args.output) / "overlays"
    qc_dir = Path(args.output) / "qc_coverage"

    if args.manifest:
        manifest = load_manifest(Path(args.manifest))
    else:
        # Build a simple manifest from the cells directory
        cda_dir = Path(args.cda_cells)
        rows = []
        for p in sorted(cda_dir.glob("*_cells.csv")):
            sid = p.name.replace("_cells.csv", "")
            rows.append({"slide_id": sid})
        if not rows:
            print(f"ERROR: no *_cells.csv found in {cda_dir}", file=sys.stderr)
            return 1
        import pandas as pd
        manifest = pd.DataFrame(rows)

    print(f"overlay: {len(manifest)} slides")
    ok = fail = 0
    qc_rows = []

    for row in iter_rows(manifest):
        sid = row["slide_id"]
        cells_path = (
            Path(row["cells_csv"])
            if row.get("cells_csv")
            else resolve_path(Path(args.cda_cells), args.cells_pattern, sid)
        )
        grid_path = (
            Path(row["tumor_grid"])
            if row.get("tumor_grid")
            else resolve_path(Path(args.tumor_grids), args.tumor_grid_pattern, sid)
        )
        output_path = resolve_path(overlay_dir, "{slide_id}_overlay.parquet", sid)

        try:
            grid = run_overlay(
                slide_id=sid,
                tumor_grid_path=grid_path,
                cells_path=cells_path,
                output_path=output_path,
            )
            from .qc import overlay_qc_summary
            qc_rows.append(overlay_qc_summary(grid, sid))
            ok += 1
        except Exception as exc:
            print(f"  WARNING [{sid}]: {exc}", file=sys.stderr)
            fail += 1

    if qc_rows:
        from .qc import write_qc_report
        write_qc_report(qc_rows, qc_dir / "overlay_coverage.csv")

    print(f"\noverlay: {ok} ok, {fail} failed/skipped")
    return 0 if fail == 0 else 1


# ---------------------------------------------------------------------------
# extract-features
# ---------------------------------------------------------------------------


def cmd_extract_features(args: argparse.Namespace) -> int:
    from .manifest import load_manifest, resolve_path
    from .features import extract_features_batch

    overlay_dir = Path(args.overlays)
    output_path = Path(args.output) / "slide_features" / "final_features.csv"

    if args.manifest:
        manifest = load_manifest(Path(args.manifest))
        slide_ids = list(manifest["slide_id"])
    else:
        slide_ids = [
            p.name.replace("_overlay.parquet", "")
            for p in sorted(overlay_dir.glob("*_overlay.parquet"))
        ]

    if not slide_ids:
        print("ERROR: no slides found", file=sys.stderr)
        return 1

    print(f"extract-features: {len(slide_ids)} slides")
    try:
        extract_features_batch(
            slide_ids=slide_ids,
            overlay_dir=overlay_dir,
            output_path=output_path,
            component_json_dir=Path(args.component_json) if args.component_json else None,
            component_pattern=args.component_pattern,
            measurements_csv=(
                Path(args.measurements_csv) if args.measurements_csv else None
            ),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# run (pipeline)
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    return run_pipeline(
        qupath_bin=args.qupath_bin,
        groovy_script=Path(args.groovy_script),
        slides_dir=Path(args.slides) if args.slides else None,
        roi_masks_dir=Path(args.roi_masks) if args.roi_masks else None,
        tumor_grids_dir=Path(args.tumor_grids) if args.tumor_grids else None,
        output_dir=Path(args.output),
        manifest_csv=Path(args.manifest) if args.manifest else None,
        slide_ext=args.slide_ext,
        roi_pattern=args.roi_pattern,
        tumor_grid_pattern=args.tumor_grid_pattern,
        export_geojson=args.export_geojson,
        force=args.force,
        dry_run=args.dry_run,
        object_classifier_path=(
            Path(args.object_classifier) if args.object_classifier else None
        ),
        pixel_classifier_path=(
            Path(args.pixel_classifier) if args.pixel_classifier else None
        ),
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_common_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", metavar="CSV",
                   help="manifest CSV with columns: slide_id, wsi_path, roi_geojson [, tumor_grid]")
    p.add_argument("--roi-masks", metavar="DIR",
                   help="directory containing ROI mask GeoJSON files")
    p.add_argument("--roi-pattern",
                   default="{slide_id}_selected_cdaroi_stage5b.geojson",
                   help="filename pattern for ROI masks (default: %(default)s)")
    p.add_argument("--output", required=True, metavar="DIR",
                   help="root output directory")


def _add_qupath_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--qupath-bin", required=True, metavar="PATH",
                   help="path to the QuPath executable")
    p.add_argument("--groovy-script", required=True, metavar="PATH",
                   help="path to scripts/run_cda_one_slide.groovy")
    p.add_argument("--object-classifier", metavar="PATH",
                   help="optional path to .json object classifier")
    p.add_argument("--pixel-classifier", metavar="PATH",
                   help="optional path to .json pixel classifier")
    p.add_argument("--export-geojson", action="store_true",
                   help="also export full detection GeoJSON (large)")
    p.add_argument("--force", action="store_true",
                   help="re-run even when output already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands without executing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsi-cda-til",
        description="QuPath/CDA TIL feature extraction for WSI tumor ROIs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run-cda ----
    p_cda = sub.add_parser("run-cda",
                            help="Launch QuPath CDA detection per slide")
    _add_common_io_args(p_cda)
    _add_qupath_args(p_cda)
    p_cda.add_argument("--slides", metavar="DIR",
                       help="directory containing WSI files")
    p_cda.add_argument("--slide-ext", default=".mrxs",
                       help="WSI file extension (default: %(default)s)")

    # ---- overlay ----
    p_ov = sub.add_parser("overlay",
                           help="Map CDA cells onto tumor-patch grids")
    p_ov.add_argument("--tumor-grids", required=True, metavar="DIR",
                      help="directory containing tumor-grid files")
    p_ov.add_argument("--cda-cells", required=True, metavar="DIR",
                      help="directory containing {slide_id}_cells.csv files")
    p_ov.add_argument("--output", required=True, metavar="DIR",
                      help="root output directory")
    p_ov.add_argument("--manifest", metavar="CSV")
    p_ov.add_argument("--tumor-grid-pattern",
                      default="{slide_id}_selected_tumorgrid_stage5b.parquet",
                      help="filename pattern for tumor grids (default: %(default)s)")
    p_ov.add_argument("--cells-pattern",
                      default="{slide_id}_cells.csv",
                      help="filename pattern for cells CSVs (default: %(default)s)")
    p_ov.add_argument("--roi-pattern",
                      default="{slide_id}_selected_cdaroi_stage5b.geojson")

    # ---- extract-features ----
    p_fe = sub.add_parser("extract-features",
                           help="Compute slide-level TIL/spatial features")
    p_fe.add_argument("--overlays", required=True, metavar="DIR",
                      help="directory containing {slide_id}_overlay.parquet files")
    p_fe.add_argument("--output", required=True, metavar="DIR",
                      help="root output directory")
    p_fe.add_argument("--manifest", metavar="CSV")
    p_fe.add_argument("--component-json", metavar="DIR",
                      help="directory containing component_selection JSON files")
    p_fe.add_argument("--component-pattern",
                      default="{slide_id}_component_selection_stage5b.json")
    p_fe.add_argument("--measurements-csv", metavar="PATH",
                      help="path to CDA measurements.csv (optional)")

    # ---- run (full pipeline) ----
    p_run = sub.add_parser("run", help="Run all stages in sequence")
    _add_common_io_args(p_run)
    _add_qupath_args(p_run)
    p_run.add_argument("--slides", metavar="DIR")
    p_run.add_argument("--slide-ext", default=".mrxs")
    p_run.add_argument("--tumor-grids", metavar="DIR",
                       help="directory containing tumor-grid files")
    p_run.add_argument("--tumor-grid-pattern",
                       default="{slide_id}_selected_tumorgrid_stage5b.parquet")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "run-cda":          cmd_run_cda,
        "overlay":          cmd_overlay,
        "extract-features": cmd_extract_features,
        "run":              cmd_run,
    }
    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)
    sys.exit(fn(args))


if __name__ == "__main__":
    main()
