"""Build and execute QuPath CLI commands for per-slide CDA cell detection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_CMD_TEMPLATE = (
    "{qupath_bin} script"
    " --image {wsi}"
    " --args slide_id={slide_id}"
    " --args roi_geojson={roi_geojson}"
    " --args cells_csv={cells_csv}"
    " --args measurements_csv={measurements_csv}"
    " --args export_geojson={export_geojson}"
    "{geojson_arg}"
    "{obj_class_arg}"
    "{pix_class_arg}"
    " {script}"
)


def build_command(
    slide_id: str,
    wsi_path: Path,
    roi_geojson: Path,
    cells_csv: Path,
    measurements_csv: Path,
    groovy_script: Path,
    qupath_bin: str,
    export_geojson: bool = False,
    out_geojson: Path | None = None,
    object_classifier_path: Path | None = None,
    pixel_classifier_path: Path | None = None,
) -> str:
    geojson_arg = (
        f" --args out_geojson={out_geojson}" if export_geojson and out_geojson else ""
    )
    obj_class_arg = (
        f" --args object_classifier_path={object_classifier_path}"
        if object_classifier_path else ""
    )
    pix_class_arg = (
        f" --args pixel_classifier_path={pixel_classifier_path}"
        if pixel_classifier_path else ""
    )
    return _CMD_TEMPLATE.format(
        qupath_bin=qupath_bin,
        wsi=wsi_path,
        slide_id=slide_id,
        roi_geojson=roi_geojson,
        cells_csv=cells_csv,
        measurements_csv=measurements_csv,
        export_geojson="true" if export_geojson else "false",
        geojson_arg=geojson_arg,
        obj_class_arg=obj_class_arg,
        pix_class_arg=pix_class_arg,
        script=groovy_script,
    )


def run_one(
    slide_id: str,
    wsi_path: Path,
    roi_geojson: Path,
    cells_csv: Path,
    measurements_csv: Path,
    groovy_script: Path,
    qupath_bin: str,
    export_geojson: bool = False,
    out_geojson: Path | None = None,
    object_classifier_path: Path | None = None,
    pixel_classifier_path: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run CDA detection for a single slide. Returns True on success."""
    success_file = out_geojson if export_geojson else cells_csv

    if not dry_run:
        if not roi_geojson.exists():
            print(
                f"  WARNING [{slide_id}]: ROI GeoJSON not found: {roi_geojson}",
                file=sys.stderr,
            )
            return False
        if success_file and success_file.exists() and measurements_csv.exists() and not force:
            print(f"  [skip] {slide_id}: output already exists")
            return True

    cells_csv.parent.mkdir(parents=True, exist_ok=True)
    measurements_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_geojson:
        out_geojson.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        slide_id=slide_id,
        wsi_path=wsi_path,
        roi_geojson=roi_geojson,
        cells_csv=cells_csv,
        measurements_csv=measurements_csv,
        groovy_script=groovy_script,
        qupath_bin=qupath_bin,
        export_geojson=export_geojson,
        out_geojson=out_geojson,
        object_classifier_path=object_classifier_path,
        pixel_classifier_path=pixel_classifier_path,
    )

    if dry_run:
        print(f"  [dry-run] {cmd}")
        return True

    print(f"  [{slide_id}] running QuPath CDA")
    rc = subprocess.run(cmd, shell=True).returncode
    if rc != 0:
        print(f"  FAILED [{slide_id}]: QuPath exit code {rc}", file=sys.stderr)
        return False

    if success_file and not success_file.exists():
        print(
            f"  WARNING [{slide_id}]: QuPath returned 0 but expected output "
            f"not found: {success_file}",
            file=sys.stderr,
        )
        return False

    return True
