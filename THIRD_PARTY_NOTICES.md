# Third-Party Notices

---

## Upstream CDA cell-detection algorithm

This project wraps and extends the CDA (Colour Deconvolution and classification
Algorithm) cell-detection pipeline originally developed by Thazin Nwe Aung.

**Repository:** https://github.com/tznaung/Mel_Color_Norm-CellDetection  
**Author:** Thazin Nwe Aung  
**Publication:**  
Aung TN, Qu Z, Kortylewski D, Bhardwaj N, Bhatt DL, Chen M, Li Y, Bhatt YM,
Balli M, Bhatt DH, Bhatt M, Bhatt A, Liu R, Bhatt Y, Bhatt S, Bhatt K, Bhatt V,
Bhatt B, Bhatt J, Bhatt N, Bhatt U.  
*EBioMedicine*, 2022. DOI: 10.1016/j.ebiom.2022.104143

The following elements of this project are derived from or adapted from the
upstream CDA repository:

- `scripts/run_cda_one_slide.groovy`: QuPath CLI wrapper adapted from
  `Cells_Calculator_ANNSEP24.groovy`. Adapted elements include:
  - Colour deconvolution stain vectors (Hematoxylin / Eosin OD values)
  - Watershed cell-detection parameters (pixel size, radii, thresholds)
  - Smoothing feature plugin parameters (FWHM values)
  - Object classifier architecture (ANN-MLP) structure and class names
    (Immune cells, Tumor, Stroma, Other)
  - Tissue-wrapping pixel classifier architecture

  **Note:** classifier asset files (trained model weights) are **not** bundled
  in this repository. Users must obtain them from the upstream CDA repository
  and supply the paths via `object_classifier_path` and `pixel_classifier_path`
  arguments. If `object_classifier_path` is omitted, the script may attempt to
  use a user-configured QuPath project classifier named `ANN_MLP_sep24`, but
  this is not provided here and should not be relied on for reproducibility.

**Upstream MIT License:**

```
MIT License

Copyright (c) 2022 Thazin Nwe Aung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## QuPath

QuPath is an open source, cross-platform software application designed for
digital pathology image analysis.

**Website:** https://qupath.github.io  
**Repository:** https://github.com/qupath/qupath  
**License:** GNU General Public License v3.0  
**Citation:**  
Bankhead P, Loughrey MB, Fernández JA, et al.  
*QuPath: Open source software for digital pathology image analysis.*  
Scientific Reports, 2017. DOI: 10.1038/s41598-017-17204-5

---

## OpenSlide (optional dependency)

If you use `openslide-python` for reading WSI bounds offsets:

**Website:** https://openslide.org  
**Repository:** https://github.com/openslide/openslide-python  
**License:** GNU Lesser General Public License v2.1

---

## Python dependencies

The following packages are used under their respective open-source licenses:

| Package       | License    | URL |
|---------------|------------|-----|
| NumPy         | BSD-3      | https://numpy.org |
| pandas        | BSD-3      | https://pandas.pydata.org |
| SciPy         | BSD-3      | https://scipy.org |
| scikit-learn  | BSD-3      | https://scikit-learn.org |
| pyarrow       | Apache-2.0 | https://arrow.apache.org |
| Shapely       | BSD-3      | https://shapely.readthedocs.io |
| GeoPandas     | BSD-3      | https://geopandas.org |

---

## What this project adds

The wsi-cda-til-features codebase adds:

1. QuPath batch automation and CLI argument handling (no embedded private paths)
2. ROI-restricted CDA execution using stage5b GeoJSON masks from
   [wsi-prototype-tumor-masker](https://github.com/Hernfe/WSI-Prototype-Tumor-Masker)
3. Cell centroid-to-tumor-patch overlay computation
4. Engineered TIL density, spatial dispersion, section, satellite, and
   component-geometry features

We did not create the CDA cell-detection algorithm. We credit and thank
Thazin Nwe Aung and co-authors for making it openly available.

---

## Limitations and research-use notice

- The underlying CDA was trained for **melanoma H&E** histology.
  It has not been validated for other tumour types.
- **Research use only.** This tool is not intended for and must not be used
  for clinical diagnosis, treatment planning, or any regulated medical
  decision-making purpose.
- Users must visually quality-control cell detections, ROI masks, and
  extracted features before drawing any scientific conclusions.
- Stroma and Other detection classes have lower precision in our usage and
  are excluded from default feature extraction.
