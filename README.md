# wsi-cda-til-features

QuPath batch automation, ROI-restricted CDA execution, cell-to-tumor-mask overlay,
and engineered TIL / spatial feature extraction for whole-slide images.

> **Research use only.** Not for clinical diagnosis, treatment planning, or any
> regulated medical decision-making.  
> **Data privacy:** never commit real WSIs, patient identifiers, or
> cohort-linked slide IDs to this repository.

---

## Overview

This tool follows the
[wsi-prototype-tumor-masker](https://github.com/Hernfe/WSI-Prototype-Tumor-Masker)
pipeline. The masker outputs two GeoJSON files per slide:

```
{slide_id}_core.geojson       — selected tumor ROI (intratumoral / TIL analysis)
{slide_id}_peritumor.geojson  — CDA analysis ROI (core + ~300 µm surrounding band)
```

**Terminology:** TIL (tumor-infiltrating leukocyte) refers only to leukocytes
inside the core tumor mask.  Immune cells outside the core but inside the
peritumoral band are called **peritumoral leukocytes** or **peritumoral immune
cells** — not TILs.

This extension then:

1. **`run-cda`** — launches QuPath with a Groovy wrapper to run CDA cell
   detection inside ``{slide_id}_peritumor.geojson``, writing compact cell
   CSVs and optional detection GeoJSONs.
2. **`overlay`** — maps CDA cell centroids onto tumor patch grids.
   Core counts come from patch-level assignment.  Total counts come from the
   cells CSV (CDA ran on the peritumor ROI).
   Ring counts = total − core.  Geometry from core/peritumor GeoJSONs is used
   for area calculations only.
3. **`extract-features`** — reads overlay outputs and region summaries, then
   writes a final ``slide_features/final_features.csv``.
4. **`run`** — convenience command that runs all three stages in sequence.

The underlying cell-detection algorithm is
**CDA** by Thazin Nwe Aung
([tznaung/Mel_Color_Norm-CellDetection](https://github.com/tznaung/Mel_Color_Norm-CellDetection),
MIT license). We did not create the CDA algorithm; we add the batch
automation and feature-extraction layer on top of it.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full attribution.

---

## Installation

```bash
pip install -e ".[dev]"          # development install with test dependencies
pip install -e ".[geojson]"      # add Shapely/GeoPandas for GeoJSON fallback loading
pip install -e ".[openslide]"    # add openslide-python for GeoJSON offset correction
```

Requires Python ≥ 3.10.

---

## Quickstart: using existing CDA cells CSVs

If you already have `{slide_id}_cells.csv` outputs from a previous CDA run,
start at the overlay step:

```bash
wsi-cda-til overlay \
  --tumor-grids  /path/to/tumor_grids/ \
  --cda-cells    /path/to/cda_cells/ \
  --output       /path/to/output/

wsi-cda-til extract-features \
  --overlays     /path/to/output/overlays/ \
  --output       /path/to/output/
```

Results are written to `/path/to/output/slide_features/final_features.csv`.

---

## Quickstart: running QuPath/Groovy CDA

### Prerequisites

- [QuPath](https://qupath.github.io) ≥ 0.5 installed and executable from CLI
- The ANN-MLP object classifier and (optionally) the pixel classifier from
  [tznaung/Mel_Color_Norm-CellDetection](https://github.com/tznaung/Mel_Color_Norm-CellDetection)
  (saved as `.json` files from your own local CDA/QuPath setup —
  **these are not bundled in this repository**)
- Tumor ROI GeoJSON masks from
  [wsi-prototype-tumor-masker](https://github.com/Hernfe/WSI-Prototype-Tumor-Masker)

### Single-slide dry-run

```bash
wsi-cda-til run-cda \
  --slides       /path/to/slides/ \
  --roi-masks    /path/to/roi_masks/   # directory containing *_peritumor.geojson files \
  --qupath-bin   $QUPATH_BIN \
  --groovy-script scripts/run_cda_one_slide.groovy \
  --object-classifier /path/to/ANN_MLP_sep24.json \
  --pixel-classifier  /path/to/pixel_classifier.json \
  --output       /path/to/output/ \
  --dry-run
```

### Manifest-based batch run

```bash
wsi-cda-til run \
  --manifest     examples/example_manifest.csv \
  --tumor-grids  /path/to/tumor_grids/ \
  --qupath-bin   $QUPATH_BIN \
  --groovy-script scripts/run_cda_one_slide.groovy \
  --output       /path/to/output/
```

---

## File naming patterns

| File | Default pattern |
|------|----------------|
| Peritumor mask (CDA input) | `{slide_id}_peritumor.geojson` |
| Core tumor mask | `{slide_id}_core.geojson` |
| Tumor grid | `{slide_id}_selected_tumorgrid_stage5b.parquet` |
| CDA cells CSV | `{slide_id}_cells.csv` |
| Overlay output | `{slide_id}_overlay.parquet` |
| Region summary | `{slide_id}_region_summary.json` |
| Component selection | `{slide_id}_component_selection_stage5b.json` |

All patterns can be overridden via `--roi-pattern`, `--core-pattern`,
`--tumor-grid-pattern`, `--cells-pattern`, and `--component-pattern` flags.

---

## Manifest CSV

You can supply a manifest CSV instead of scanning directories:

```csv
slide_id,wsi_path,roi_geojson,tumor_grid
example_slide_001,/slides/example_slide_001.mrxs,/masks/example_slide_001_peritumor.geojson,/grids/example_slide_001_tumorgrid.parquet
```

Required column: `slide_id`.  
Optional columns: `wsi_path`, `roi_geojson`, `tumor_grid`, `cells_csv`.

See [examples/example_manifest.csv](examples/example_manifest.csv).

---

## Output features

The `extract-features` command produces one row per slide in `final_features.csv`:

| Feature | Description |
|---------|-------------|
| `slide_id` | Slide identifier |
| `wsi_filename` | WSI filename if available |
| `patch_count_total` | Total patches in overlay |
| `patch_count_tumor` | Patches classified as tumor |
| `patch_count_peritumor` | Non-tumor patches within dilation radius |
| `tumor_area_mm2` | Tumor area (patch count × patch area) |
| `total_cda_immune_cells` | Total immune cells across slide |
| `total_cda_tumor_cells` | Total tumor cells across slide |
| `mean_intratumoral_til_count_per_patch` | Mean TIL count per tumor patch (core only) |
| `max_intratumoral_til_count_per_patch` | Max TIL count in a single tumor patch |
| `mean_intratumoral_til_density_per_mm2` | Mean TIL density (core patches, cells/mm²) |
| `max_intratumoral_til_density_per_mm2` | Max TIL density (core patches) |
| `mean_tumor_cell_density_per_mm2` | Mean tumor cell density (core patches) |
| `fraction_tumor_patches_til_count_ge_5` | Fraction of core patches with ≥5 TILs |
| `fraction_tumor_patches_til_count_ge_15` | Fraction with ≥15 |
| `fraction_tumor_patches_til_count_ge_30` | Fraction with ≥30 |
| `peritumoral_leukocyte_count` | Immune cells in the peritumoral ring (total minus core; NaN if no region summary) |
| `peritumoral_leukocyte_density_per_mm2` | Peritumoral leukocyte density (cells/mm²; NaN if absent) |
| `peritumoral_immune_cell_count` | Alias for peritumoral_leukocyte_count |
| `peritumoral_immune_cell_density_per_mm2` | Alias for peritumoral_leukocyte_density_per_mm2 |
| `peritumoral_ring_area_mm2` | Ring area (peritumor minus core, mm²; NaN if absent) |
| `peritumoral_ring_patch_count` | Approximate patch count in the ring |

> **Note:** "TIL" (tumor-infiltrating leukocyte) refers only to leukocytes
> inside the core tumor mask.  Immune cells in the peritumoral ring are called
> peritumoral leukocytes or peritumoral immune cells, not TILs.
| `n_sections` | Number of DBSCAN-detected tumor tissue sections |
| `multi_section_flag` | True when >1 section detected |
| `max_inter_section_distance_px` | Max centroid distance between sections |
| `secondary_sections_area_mm2` | Combined area of non-primary sections |
| `primary_n_components` | 8-connected components in primary section |
| `primary_section_n_cc_satellites` | Satellite connected components |
| `primary_main_component_patches` | Patch count of largest component |
| `primary_main_component_area_mm2` | Area of largest component |
| `primary_satellite_area_mm2` | Total satellite area |
| `primary_satellite_area_fraction` | Satellite fraction of primary section |
| `primary_mean_satellite_distance_px` | Mean satellite–main centroid distance |
| `primary_max_satellite_distance_px` | Max satellite–main centroid distance |
| `primary_section_til_dispersion_mean_distance_px` | Weighted mean TIL distance from TIL centroid |
| `primary_section_til_dispersion_max_distance_px` | Max TIL distance from TIL centroid |

Component-selection metadata (when `--component-json` is provided):

| Feature | Description |
|---------|-------------|
| `sel_uncertainty` | 1 if stage5b flagged selection uncertainty |
| `sel_n_satellite_ccs` | Number of selected satellite connected components |
| `sel_group_median_margin` | Median classifier margin for selected group |
| `sel_group_high_conf_frac` | Fraction of high-confidence patches in group |
| `sel_group_compactness` | Group compactness score |
| `sel_stage5_area_mm2` | Total area at stage5b selection |

---

## Limitations

- The underlying CDA was **trained for melanoma H&E** histology.
  Results on other tissue types or staining protocols are unvalidated.
- **Research use only** — not for clinical diagnosis, treatment planning,
  or regulated medical decision-making.
- Users must visually quality-control cell detections, ROI masks, and
  extracted features before drawing any scientific conclusions.
- Stroma and Other CDA classes have lower precision in our usage;
  they are excluded from default feature extraction.
- Peritumoral leukocyte density is NaN when no region summary JSON was produced
  by the overlay step (e.g. when core/peritumor GeoJSON paths were not provided).

---

## Credits and citation

**CDA cell-detection algorithm:**  
Aung TN, Qu Z, Kortylewski D, Bhardwaj N, Bhatt DL, et al.  
*EBioMedicine*, 2022. DOI: [10.1016/j.ebiom.2022.104143](https://doi.org/10.1016/j.ebiom.2022.104143)

**QuPath:**  
Bankhead P, Loughrey MB, Fernández JA, et al.  
*QuPath: Open source software for digital pathology image analysis.*  
Scientific Reports, 2017. DOI: [10.1038/s41598-017-17204-5](https://doi.org/10.1038/s41598-017-17204-5)

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full licence text
and dependency attributions.

---

## License

This wrapper / feature-extraction code is released under the
[MIT License](LICENSE).

The upstream CDA algorithm is separately MIT licensed by Thazin Nwe Aung.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
