/* groovylint-disable DuplicateStringLiteral, UnnecessaryGetter */
/*
QuPath-Script to detect objects using StarDist and export detection objects to
.csv files (e.g. to be used for Classification)
*/

import static qupath.lib.gui.scripting.QPEx.*

import java.awt.image.BufferedImage
import qupath.ext.stardist.StarDist2D
import qupath.lib.gui.tools.MeasurementExporter
import qupath.lib.projects.ProjectIO
import qupath.lib.projects.ProjectImageEntry
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathDetectionObject

// Specify the project path
/* groovylint-disable-next-line LineLength */
PROJECT_PATH = /D:\Scans Iris HJS\ebenen_schnitte\Classification_2021_02_07\project.qpproj/
// Specify the model .pb file (you will need to change this!)
STARDIST_MODELPATH = '../models_stardist/he_heavy_augment.pb'
// Enter the output root and directory name
OUT_ROOT = /..\data/
OUT_DIR = 'nuclei_stardist_2021-11-11'

EXPORT = true

main()
print 'Script finished.'

void main() {
    // Main function of this script
    if (PROJECT_PATH) {
        print("Using project path: $PROJECT_PATH")
        // projectFile = new File(pathname = PROJECT_PATH)
        /* project = ProjectIO.loadProject(
            fileProject = projectFile, cls = BufferedImage.class
            )
        */
    }
    print('Using open project.')
    project = getProject()

    List trainingImagesNames = []
    for (String i in 1..25) {
        trainingImagesNames.add(i)
    }

    List trainingImagesList = []
    for (image in project.getImageList()) {
        if (image.getImageName() in trainingImagesNames) {
            trainingImagesList.add(image)
        }
    }

    List imagesToExport = []
    for (image in trainingImagesList) {
        // detect(image)
        imagesToExport.add(image)
    }
    if (!imagesToExport.isEmpty() && EXPORT) {
        export(imageList = imagesToExport, group = 'Training')
    }
}

void detect(ProjectImageEntry image) {
    // This is where the cell detection using StarDist starts
    print 'RUNNING DETECTION FOR IMAGE: &image.getImageName()'

    stardist = StarDist2D.builder(STARDIST_MODELPATH)
        .threshold(0.5)              // Prediction threshold
        .normalizePercentiles(1, 99) // Percentile normalization
        .pixelSize(0.221)              // Resolution for detection
        //.tileSize(1024)              // Specify width & height of the tile used for prediction
        .cellExpansion(5.0)          // Approximate cells based upon nucleus expansion
        //.cellConstrainScale(1.5)     // Constrain cell expansion using nucleus size
        .ignoreCellOverlaps(true)   // Set to true if you don't care if cells expand into one another
        .measureShape()              // Add shape me asurements
        .measureIntensity()          // Add cell measurements (in all compartments)
        .includeProbability(true)      // Add probability as a measurement (enables later filtering)
        //.simplify(1)                 // Control how polygons are 'simplified' to remove unnecessary vertices
        //.doLog()                     // Use this to log a bit more information while running the script
        //.createAnnotations()         // Generate annotation objects using StarDist, rather than detection objects
        .constrainToParent(false)      // Prevent nuclei/cells expanding beyond any parent annotations (default is true)
        //.classify('Tumor')           // Automatically assign all created objects as 'Tumor'
        .build()

    // First delete all existing detections
    imageData = image.readImageData()
    hierarchy = imageData.getHierarchy()
    oldDetections = hierarchy.getDetectionObjects()
    if (oldDetections.size() > 0) {
        print('Old detections are being cleared: ' + oldDetections.size())
        hierarchy.removeObjects(
            pathObjects = oldDetections, keepChildren = false
            )
        fireHierarchyUpdate(hierarchy)
        print('Current number of  detections: ' + hierarchy.getDetectionObjects().size())
    }

    // Select relevant annotations and run detection
    predicate = {
        it.isAnnotation() && (it.getPathClass() == getPathClass('SiHa')
        || it.getPathClass() == getPathClass('Keratinocytes')
        || it.getPathClass() == getPathClass('Apoptosis/Necrosis'))
    }

    pathObjects = getObjects(hierarchy, predicate)
    if (pathObjects.isEmpty()) {
        Dialogs.showErrorMessage('StarDist', 'Please select a parent object!')
        return
    }

    print('Now detecting new objects.')
    stardist.detectObjects(imageData, pathObjects)

    // This is necessary to update the image hierarchy.
    fireHierarchyUpdate(hierarchy)

    detections = hierarchy.getDetectionObjects()
    print('Number of  detections: ' + detections.size())

    detections.each { it.setPathClass(it.getParent().getPathClass()) }

    // This is necessary to update the image hierarchy.
    fireHierarchyUpdate(hierarchy)

    image.saveImageData(imageData)
}

void export(List imageList, String group) {
    // Method to export measurements
    // Separate each measurement value in the output file with a comma (.csv file)

    String separator = ','
    for (datatype in ['detections', 'annotations']) {
        if (datatype == 'detections') {
            print('Exporting detections...')
            exportType = PathDetectionObject
        }
        else if (datatype == 'annotations') {
            print('Exporting annotations...')
            exportType = PathAnnotationObject
        }
        else {
            print 'ERROR'
        }

        // Make directory if necessary
        outputDir = buildFilePath(OUT_ROOT, OUT_DIR, group)
        new File(outputDir).mkdirs()

        // Choose your *full* output path
        outputPath = buildFilePath(OUT_ROOT, OUT_DIR, group, datatype + '.csv')
        outputFile = new File(outputPath)

        // Create the measurementExporter and start the export
        new MeasurementExporter()
        .imageList(imageList)            // Images from which measurements will be exported
        .separator(separator)                 // Character that separates values
        //.includeOnlyColumns(columnsToInclude) // Columns are case-sensitive
        .exportType(exportType)               // Type of objects to export
        .exportMeasurements(outputFile)        // Start the export process
    }
    print 'Export Done!'
}
