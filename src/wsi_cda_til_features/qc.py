"""QC checks and coverage summaries for overlay outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def overlay_qc_summary(overlay: pd.DataFrame, slide_id: str = "") -> dict:
    """Compute basic QC statistics for an overlay DataFrame."""
    n_total = len(overlay)
    n_tumor = int(overlay["is_tumor"].sum()) if "is_tumor" in overlay.columns else 0
    n_peri = (
        int(overlay["is_peritumor_patch"].sum())
        if "is_peritumor_patch" in overlay.columns
        else 0
    )
    n_immune = (
        int(overlay["cda_immune_count"].sum())
        if "cda_immune_count" in overlay.columns
        else 0
    )
    n_tumor_cells = (
        int(overlay["cda_tumor_cell_count"].sum())
        if "cda_tumor_cell_count" in overlay.columns
        else 0
    )
    tumor_patches_with_cells = (
        int((overlay.loc[overlay["is_tumor"], "cda_immune_count"] > 0).sum())
        if "cda_immune_count" in overlay.columns and n_tumor > 0
        else 0
    )

    return {
        "slide_id": slide_id,
        "n_total_patches": n_total,
        "n_tumor_patches": n_tumor,
        "n_peritumor_patches": n_peri,
        "n_immune_cells": n_immune,
        "n_tumor_cells": n_tumor_cells,
        "tumor_patches_with_immune_cells": tumor_patches_with_cells,
        "tumor_coverage_fraction": (
            tumor_patches_with_cells / n_tumor if n_tumor > 0 else float("nan")
        ),
    }


def write_qc_report(rows: list[dict], output_path: Path) -> None:
    """Write QC summary rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  QC report: {output_path}")
