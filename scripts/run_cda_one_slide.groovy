// run_cda_one_slide.groovy
//
// NOTICE: This script is adapted as a QuPath CLI wrapper around the CDA cell
// detection logic from:
//
//   tznaung/Mel_Color_Norm-CellDetection
//   Thazin Nwe Aung — MIT License
//   Aung et al., EBioMedicine 2022. DOI: 10.1016/j.ebiom.2022.104143
//
// The upstream cell-detection pipeline (watershed detection, colour
// deconvolution parameters, smoothing radii, and object-classification
// architecture) is reproduced here as a QuPath batch CLI wrapper.
// See THIRD_PARTY_NOTICES.md for the full upstream MIT licence notice.
//
// This wrapper adds:
//   - key=value CLI argument parsing (QuPath --args)
//   - ROI import from a stage5b GeoJSON mask (output of wsi-prototype-tumor-masker)
//   - compact cells CSV export (cell_x, cell_y, cell_class)
//   - optional detection GeoJSON export
//   - image-level measurement CSV append
//
// All paths come from key=value args — no private paths are hard-coded.
//
// Required args (pass via QuPath --args key=value):
//   slide_id           slide identifier string
//   roi_geojson        path to stage5b CDA ROI GeoJSON (input)
//   measurements_csv   path to append image-level measurements CSV (output)
//
// Optional args:
//   cells_csv               path to write compact cells CSV (output; recommended)
//   export_geojson          true/false (default: false)
//   out_geojson             path for detection GeoJSON (only used when export_geojson=true)
//   object_classifier_path  path to .json ANN-MLP object classifier
//                           (obtain from tznaung/Mel_Color_Norm-CellDetection)
//   pixel_classifier_path   path to .json pixel classifier for tissue trimming
//                           (obtain from tznaung/Mel_Color_Norm-CellDetection)
//
// NOTE: Classifier assets are not bundled in this repository.
// Provide object_classifier_path (and optionally pixel_classifier_path) from
// your own local CDA/QuPath setup (obtain from tznaung/Mel_Color_Norm-CellDetection).
// If object_classifier_path is omitted, the script may attempt to use a
// user-configured QuPath project classifier named ANN_MLP_sep24, but this
// classifier is not bundled here and should not be relied on for reproducibility.
//
// Example invocation:
//   QuPath script \
//     --image /path/to/slide.mrxs \
//     --args slide_id=example_slide_001 \
//     --args roi_geojson=/output/roi_masks/example_slide_001_selected_cdaroi_stage5b.geojson \
//     --args cells_csv=/output/cda/example_slide_001_cells.csv \
//     --args measurements_csv=/output/cda/measurements.csv \
//     --args export_geojson=false \
//     scripts/run_cda_one_slide.groovy

import qupath.lib.color.ColorDeconvolutionStains
import qupath.lib.color.StainVector
import qupath.lib.io.GsonTools
import qupath.lib.io.PathIO
import java.io.File
import java.io.FileWriter
import java.io.BufferedWriter
import java.io.BufferedReader
import java.io.FileReader

// ---- Parse args (key=value) ----
def argMap = [:]
if (args != null) {
    args.each { arg ->
        def idx = arg.indexOf('=')
        if (idx > 0) {
            argMap[arg.substring(0, idx).trim()] = arg.substring(idx + 1).trim()
        }
    }
}

def slideId          = argMap["slide_id"]
def roiGeojson       = argMap["roi_geojson"]
def outGeojson       = argMap.getOrDefault("out_geojson", "")
def measurementsCsv  = argMap["measurements_csv"]
def cellsCsv         = argMap.getOrDefault("cells_csv", "")
def exportGeojson    = argMap.getOrDefault("export_geojson", "false").toLowerCase() == "true"
def objectClassPath  = argMap.getOrDefault("object_classifier_path", "")
def pixelClassPath   = argMap.getOrDefault("pixel_classifier_path", "")

if (!slideId)        { println "ERROR: slide_id arg required";        return }
if (!roiGeojson)     { println "ERROR: roi_geojson arg required";     return }
if (!measurementsCsv){ println "ERROR: measurements_csv arg required";return }

println "=== CDA: starting for ${slideId} ==="

// ---- Check ROI GeoJSON ----
def roiFile = new File(roiGeojson)
if (!roiFile.exists()) {
    println "ERROR: roi_geojson not found: ${roiGeojson}"
    return
}

// ---- Import stage5b ROI as annotations ----
println "Importing ROI from ${roiGeojson}"
def importedAnnotations = PathIO.readObjects(roiFile)
if (importedAnnotations == null || importedAnnotations.isEmpty()) {
    println "ERROR: no objects could be read from ${roiGeojson}"
    return
}
clearAnnotations()
addObjects(importedAnnotations)
fireHierarchyUpdate()
println "Imported ${importedAnnotations.size()} annotation(s)"

// ---- Settings (matching Cells_Calculator_ANNSEP24.groovy from upstream CDA) ----
List<Double> smoothing = [25, 50]

// ---- Set image type (required before setting stains) ----
import qupath.lib.images.ImageData as ImageDataClass
getCurrentImageData().setImageType(ImageDataClass.ImageType.BRIGHTFIELD_H_E)

// ---- Set pixel size ----
setPixelSizeMicrons(getCurrentImageData(), 0.2500, 0.2500, 1)

// ---- Set stain vectors (from upstream CDA paper) ----
double[] hematoxylinRGB = [0.651, 0.701, 0.29]
double[] eosinRGB       = [0.216, 0.801, 0.558]
double[] backgroundRGB  = [255, 255, 255]
StainVector customHematoxylin = StainVector.createStainVector(
    "Hematoxylin", hematoxylinRGB[0], hematoxylinRGB[1], hematoxylinRGB[2])
StainVector customEosin = StainVector.createStainVector(
    "Eosin", eosinRGB[0], eosinRGB[1], eosinRGB[2])
ColorDeconvolutionStains cds = new ColorDeconvolutionStains(
    "CustomCDS", customHematoxylin, customEosin,
    backgroundRGB[0], backgroundRGB[1], backgroundRGB[2])
getCurrentImageData().setColorDeconvolutionStains(cds)

clearDetections()
selectAnnotations()

def imageData = getCurrentImageData()
def server    = imageData.getServer()
def pixelSize = server.getPixelCalibration().getPixelHeightMicrons()
def hierarchy = getCurrentHierarchy()

// ---- Partition imported annotations into ROI and Ignore regions ----
List<PathObject> roiAnnotations    = []
List<PathObject> ignoreAnnotations = []
getAnnotationObjects().each { ann ->
    if (ann.getPathClass() == getPathClass("Ignore*")) {
        ignoreAnnotations << ann
    } else {
        ann.setPathClass(getPathClass("Temp"))
        roiAnnotations << ann
    }
}

mergeAnnotations(roiAnnotations)
mergeAnnotations(ignoreAnnotations)

PathObject roiAnnotation    = null
PathObject ignoreAnnotation = null
getAnnotationObjects().each { ann ->
    if (ann.getPathClass() == getPathClass("Ignore*")) {
        ignoreAnnotation = ann
    } else {
        roiAnnotation = ann
    }
}

if (roiAnnotation == null) {
    println "ERROR: no non-Ignore annotation found after import"
    return
}

// ---- Subtract ignore regions and wrap to tissue boundary ----
double roiAnnotationArea
try {
    roiAnnotation = subtractAnnotations(roiAnnotation, ignoreAnnotation)
    roiAnnotation = wrapAnnotation(roiAnnotation, pixelClassPath)
    roiAnnotationArea = roiAnnotation.getROI().getArea()
} catch (GroovyRuntimeException | NullPointerException e) {
    println "ERROR in annotation processing: ${e}"
    return
}

// ---- Cell detection (watershed, matching upstream CDA parameters) ----
selectObjects(roiAnnotation)
println "Running watershed cell detection..."
runPlugin(
    'qupath.imagej.detect.cells.WatershedCellDetection',
    '{"detectionImageBrightfield": "Hematoxylin OD",' +
    ' "requestedPixelSizeMicrons": 0.5,' +
    ' "backgroundRadiusMicrons": 8.0,' +
    ' "medianRadiusMicrons": 0.0,' +
    ' "sigmaMicrons": 1.5,' +
    ' "minAreaMicrons": 10.0,' +
    ' "maxAreaMicrons": 400.0,' +
    ' "threshold": 0.1,' +
    ' "maxBackground": 2.0,' +
    ' "watershedPostProcess": true,' +
    ' "cellExpansionMicrons": 5.0,' +
    ' "includeNuclei": true,' +
    ' "smoothBoundaries": false,' +
    ' "makeMeasurements": true}'
)

println "Smoothing features..."
for (double sv : smoothing) {
    runPlugin(
        'qupath.lib.plugins.objects.SmoothFeaturesPlugin',
        '{"fwhmMicrons": ' + sv + ', "smoothWithinClasses": false}'
    )
}

// ---- Object classification (ANN-MLP from upstream CDA) ----
println "Running object classification..."
doRunObjectClassifier(objectClassPath)

// ---- Count detections by class ----
Map immuneMap  = [pathClass: getPathClass("Immune cells"), count: 0, area: 0.0]
Map otherMap   = [pathClass: getPathClass("Other"),        count: 0, area: 0.0]
Map stromaMap  = [pathClass: getPathClass("Stroma"),       count: 0, area: 0.0]
Map tumorMap   = [pathClass: getPathClass("Tumor"),        count: 0, area: 0.0]
List<Map> pathClassList = [immuneMap, otherMap, stromaMap, tumorMap]

getDetectionObjects().each { det ->
    if      (det.getPathClass() == immuneMap.pathClass)  { immuneMap.count++;  immuneMap.area  += det.getROI().getArea() }
    else if (det.getPathClass() == otherMap.pathClass)   { otherMap.count++;   otherMap.area   += det.getROI().getArea() }
    else if (det.getPathClass() == stromaMap.pathClass)  { stromaMap.count++;  stromaMap.area  += det.getROI().getArea() }
    else                                                  { tumorMap.count++;   tumorMap.area   += det.getROI().getArea() }
}

println "Cells: Immune=${immuneMap.count}, Tumor=${tumorMap.count}, Stroma=${stromaMap.count}, Other=${otherMap.count}"

// ---- Compute image-level measurements ----
PathObject rootObject = hierarchy.getRootObject()
def measList = rootObject.getMeasurementList()

pathClassList.each { m ->
    measList.put(m.pathClass.getName() + " total area px",   m.area)
    measList.put(m.pathClass.getName() + " total area µm^2", m.area * pixelSize * pixelSize)
    measList.put(m.pathClass.getName() + " total area %",    m.area / roiAnnotationArea * 100)
}
measList.put("Total ROI area px",       roiAnnotationArea)
measList.put("Total ROI area µm^2", roiAnnotationArea * pixelSize * pixelSize)

try { measList.put("Total eTILs %",    (double)immuneMap.count / (tumorMap.count + immuneMap.count) * 100) }
catch (ArithmeticException ae) { println "ERROR ADDING Total eTILs %: " + ae }
try { measList.put("Total etTILs %",   (double)immuneMap.count / (immuneMap.count + tumorMap.count + stromaMap.count + otherMap.count) * 100) }
catch (ArithmeticException ae) { println "ERROR ADDING Total etTILs %: " + ae }
try { measList.put("Total esTILs %",   (double)immuneMap.count / (immuneMap.count + stromaMap.count + otherMap.count) * 100) }
catch (ArithmeticException ae) { println "ERROR ADDING Total esTILs %: " + ae }
try { measList.put("Total eaTILs mm^2", (double)immuneMap.count / (roiAnnotationArea * pixelSize * pixelSize) * 1000 * 1000) }
catch (ArithmeticException ae) { println "ERROR ADDING Total eaTILs mm^2: " + ae }
try { measList.put("Total easTILs %",  (double)immuneMap.area / (roiAnnotationArea - tumorMap.area) * 100) }
catch (ArithmeticException ae) { println "ERROR ADDING Total easTILs %: " + ae }

// ---- Export compact cells CSV (Tumor and Immune only) ----
if (cellsCsv) {
    println "Writing cells CSV to ${cellsCsv}"
    def cellsFile = new File(cellsCsv)
    cellsFile.parentFile?.mkdirs()
    def sb = new StringBuilder()
    sb.append("slide_id,cell_x,cell_y,raw_class,cell_class,cell_area_px,cell_area_um2\n")
    getDetectionObjects().each { det ->
        def rawClass = det.getPathClass()?.getName() ?: ""
        if (rawClass != "Tumor" && rawClass != "Immune cells") return
        def cellClass = (rawClass == "Tumor") ? "tumor_cell" : "immune"
        def roi = det.getROI()
        def cx  = roi.getCentroidX()
        def cy  = roi.getCentroidY()
        def areaPx  = roi.getArea()
        def areaUm2 = areaPx * pixelSize * pixelSize
        sb.append("${slideId},${cx},${cy},${rawClass},${cellClass},${areaPx},${areaUm2}\n")
    }
    cellsFile.text = sb.toString()
    println "Cells CSV written: ${getDetectionObjects().count { d -> d.getPathClass()?.getName() in ['Tumor','Immune cells'] }} rows"
}

// ---- Export detection GeoJSON (optional) ----
if (exportGeojson && outGeojson) {
    println "Exporting ${getDetectionObjects().size()} detections to ${outGeojson}"
    def outGeoFile = new File(outGeojson)
    outGeoFile.parentFile?.mkdirs()
    def fw = new BufferedWriter(new FileWriter(outGeoFile))
    def gson = GsonTools.getInstance(true)
    fw.write(gson.toJson(GsonTools.wrapFeatureCollection(getDetectionObjects())))
    fw.close()
    println "GeoJSON export done."
} else if (!exportGeojson) {
    println "GeoJSON export skipped (export_geojson=false)."
}

// ---- Append image-level measurements to CSV ----
println "Writing measurements to ${measurementsCsv}"
def measFile   = new File(measurementsCsv)
measFile.parentFile?.mkdirs()

def totalCells   = immuneMap.count + tumorMap.count + stromaMap.count + otherMap.count
def nonTumorArea = roiAnnotationArea - tumorMap.area

def csvHeaders = [
    "immune_count", "other_count", "stroma_count", "tumor_count",
    "total_cells",
    "immune_area_px", "other_area_px", "stroma_area_px", "tumor_area_px",
    "total_roi_area_px",
    "eTILs_pct", "etTILs_pct", "esTILs_pct",
    "eaTILs_per_mm2",
]
def eTILs_pct  = totalCells > 0 ? (double)immuneMap.count / (tumorMap.count + immuneMap.count) * 100 : Double.NaN
def etTILs_pct = totalCells > 0 ? (double)immuneMap.count / totalCells * 100 : Double.NaN
def esTILs_pct = (immuneMap.count + stromaMap.count + otherMap.count) > 0
    ? (double)immuneMap.count / (immuneMap.count + stromaMap.count + otherMap.count) * 100
    : Double.NaN
def eaTILs = (roiAnnotationArea * pixelSize * pixelSize) > 0
    ? (double)immuneMap.count / (roiAnnotationArea * pixelSize * pixelSize) * 1_000_000
    : Double.NaN

def csvValues = [
    immuneMap.count, otherMap.count, stromaMap.count, tumorMap.count,
    totalCells,
    immuneMap.area, otherMap.area, stromaMap.area, tumorMap.area,
    roiAnnotationArea,
    eTILs_pct, etTILs_pct, esTILs_pct, eaTILs,
]

def headerRow = (["slide_id"] + csvHeaders).join(",")
def dataRow   = ([slideId] + csvValues.collect { v -> String.valueOf(v) }).join(",")

if (!measFile.exists()) {
    measFile.text = headerRow + "\n" + dataRow + "\n"
} else {
    def firstLine = measFile.withReader { r -> r.readLine() }
    if (firstLine != headerRow) {
        println "WARNING: measurements CSV header mismatch; appending row anyway"
    }
    measFile.append(dataRow + "\n")
}

println "=== CDA: done for ${slideId} ==="

// ===========================================================================
// Helper functions
// ===========================================================================

/**
 * Trim an annotation to tissue areas using a pixel classifier.
 *
 * If pixelClassifierPath is empty the function looks for the classifier
 * embedded in the QuPath project.  Providing an explicit path from the
 * upstream CDA repository is recommended for reproducibility.
 */
static PathObject wrapAnnotation(PathObject annotation, String pixelClassifierPath) {
    if (annotation == null) return annotation

    def imageData = getCurrentImageData()

    if (pixelClassifierPath != null && pixelClassifierPath != "") {
        def classifierFile = new File(pixelClassifierPath)
        if (!classifierFile.exists()) {
            println "WARNING: pixel_classifier_path not found: ${pixelClassifierPath}; skipping tissue wrapping"
            return annotation
        }
        def classifier = qupath.lib.classifiers.pixel.PixelClassifiers.readClassifier(classifierFile)
        return qupath.lib.processing.OpenCVTools.createAnnotationsFromPixelClassifier(
            imageData, classifier, annotation, "Positive", 0)
    }

    // No classifier provided — return annotation unchanged with a warning
    println "WARNING: pixel_classifier_path not supplied; tissue wrapping skipped."
    println "         For best results, provide the pixel classifier from:"
    println "         https://github.com/tznaung/Mel_Color_Norm-CellDetection"
    return annotation
}

/**
 * Run object classification with the provided classifier path,
 * or fall back to the project's default classifier when the path is empty.
 *
 * Obtain the ANN-MLP object classifier from:
 *   https://github.com/tznaung/Mel_Color_Norm-CellDetection
 */
static void doRunObjectClassifier(String objectClassifierPath) {
    if (objectClassifierPath != null && objectClassifierPath != "") {
        def classifierFile = new File(objectClassifierPath)
        if (!classifierFile.exists()) {
            println "WARNING: object_classifier_path not found: ${objectClassifierPath}"
            println "         Falling back to project classifier (if any)"
        } else {
            runObjectClassifier(classifierFile.toPath())
            return
        }
    }
    // Attempt project classifier; log guidance if not present
    try {
        runObjectClassifier("ANN_MLP_sep24")
    } catch (Exception e) {
        println "WARNING: could not load object classifier 'ANN_MLP_sep24': ${e.message}"
        println "         Provide --args object_classifier_path=<path> or add the classifier"
        println "         to your QuPath project from:"
        println "         https://github.com/tznaung/Mel_Color_Norm-CellDetection"
    }
}

/**
 * Subtract ignoreAnnotation geometry from roiAnnotation.
 * Returns roiAnnotation unchanged when ignoreAnnotation is null.
 */
static PathObject subtractAnnotations(PathObject roiAnn, PathObject ignoreAnn) {
    if (ignoreAnn == null) return roiAnn
    import org.locationtech.jts.geom.Geometry
    def roiGeom    = roiAnn.getROI().getGeometry()
    def ignoreGeom = ignoreAnn.getROI().getGeometry()
    def subtracted = roiGeom.difference(ignoreGeom)
    def newROI     = qupath.lib.roi.GeometryTools.geometryToROI(subtracted,
                         roiAnn.getROI().getImagePlane())
    return qupath.lib.objects.PathObjects.createAnnotationObject(newROI, roiAnn.getPathClass())
}
