/*
QuPath-Script to detect objects using StarDist and export detection objects to
.csv files (e.g. to be used for Classification)
*/

import static qupath.lib.gui.scripting.QPEx.*

import qupath.ext.stardist.StarDist2D

import qupath.lib.gui.tools.MeasurementExporter
import qupath.lib.projects.ProjectIO
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathDetectionObject


// Specify the project path
PROJECT_PATH = /.\Annotations_EP\Project_Files\project.qpproj/
// Specify the model .pb file (you will need to change this!)
STARDIST_MODELPATH = "../models_stardist_stardist/he_heavy_augment.pb"
// Enter the output root and directory name
OUT_ROOT = /..\data/
OUT_DIR = "nuclei_stardist_2022-01-23"
//
GROUP = "Results" // Options: "Training", "Results"
print("GROUP: " + GROUP)
EXPORT = true
OUTPUT_FILETYPE = "both" // Options: "json", "csv", "both"

main()
print "Script finished."

void main() {
    // Main function of this script
    projectFile = new File(pathname = PROJECT_PATH)
    if (projectFile.exists()) {
        print("Loading Project from file...")
        project = ProjectIO.loadProject(fileProject = projectFile, cls = BufferedImage.class)
    } else {
        project = getProject()
    }

    if (!["Training", "Results"].contains(GROUP)) {
        throw new Exception("The entered value for `GROUP` is invalid. Allowed: [`Training`, `Results`]. Got: ${group}.")
    }
    imageList = project.getImageList()
    for (image in imageList) {
        group_label = image.getMetadataValue(key = "Group")
        if ((GROUP == "Training") && (group_label == "TrainingAnnotated")) {
            print("GROUP: " + GROUP)
            detect(image, GROUP)
            // Now export if wished
            if (EXPORT) {
                export(image = image, group = GROUP, outputFiletype = OUTPUT_FILETYPE)
            }
        } else if ((GROUP == "Results") && (group_label == "Results")) { // && (image.getImageName() = "V19-18;#287;Ebene D - 2021-01-29 13.32.43.ndpi" || image.getImageName() = "V19-18;#336;Ebene E - 2021-01-29 14.29.20.ndpi")) {
            print(image.getImageName())
            detect(image, GROUP)
            // Now export if wished
            if (EXPORT) {
                export(image = image, group = GROUP, outputFiletype = OUTPUT_FILETYPE)
            }
        }
    }
}

def detect(image, group) {
    // This is where the cell detection using StarDist starts
    print("Running detection for image from the ${group} group: " + image.getImageName())

    stardist = StarDist2D.builder(STARDIST_MODELPATH)
        .threshold(0.5)              // Prediction threshold
        .normalizePercentiles(1, 99) // Percentile normalization
        .pixelSize(0.221)              // Resolution for detection
        //.tileSize(1024)              // Specify width & height of the tile used for prediction
        .cellExpansion(5.0)          // Approximate cells based upon nucleus expansion
        //.cellConstrainScale(1.5)     // Constrain cell expansion using nucleus size
        .ignoreCellOverlaps(true)   // Set to true if you don"t care if cells expand into one another
        .measureShape()              // Add shape me asurements
        .measureIntensity()          // Add cell measurements (in all compartments)
        .includeProbability(true)      // Add probability as a measurement (enables later filtering)
        //.simplify(1)                 // Control how polygons are "simplified" to remove unnecessary vertices
        //.doLog()                     // Use this to log a bit more information while running the script
        //.createAnnotations()         // Generate annotation objects using StarDist, rather than detection objects
        .constrainToParent(false)      // Prevent nuclei/cells expanding beyond any parent annotations (default is true)
        //.classify("Tumor")           // Automatically assign all created objects as "Tumor"
        .build()

    imageData = image.readImageData()
    hierarchy = imageData.getHierarchy()

    // First delete all existing detections
    oldDetections = hierarchy.getDetectionObjects()
    if (oldDetections.size() > 0) {
        print ("Old detections are being cleared: " + oldDetections.size())
        hierarchy.removeObjects(pathObjects = oldDetections, keepChildren = false)
        fireHierarchyUpdate(hierarchy)
        print ("Current number of  detections: " + hierarchy.getDetectionObjects().size())
    }

    print("GROUP: " + group)
    if (group == "Training") {
        // Select annotations to process
        predicate = { it.isAnnotation() && (it.getPathClass() == getPathClass("SiHa") || it.getPathClass() == getPathClass("Keratinocytes") || it.getPathClass() == getPathClass("Apoptosis/Necrosis")) }
    } else if (group == "Results") {
        // Select annotations to process
        predicate = { it.isAnnotation() && (it.getPathClass() != getPathClass("ColorDeconvolution")) }
    }
    else {
    print("NO GROUP FOUND")
    }

    def pathObjects = getObjects(hierarchy=hierarchy, predicate=predicate)
    print("Number of annotations found: ${pathObjects.size()}")
    if (pathObjects.isEmpty()) {
        throw new Exception("No annotations found!")
    }

    print("Now detecting new objects.")
    stardist.detectObjects(imageData, pathObjects)

    // This is necessary to update the image hierarchy.
    fireHierarchyUpdate(hierarchy)

    print("Number of  detections: " + hierarchy.getDetectionObjects().size())

    /*
    toDelete = hierarchy.getDetectionObjects().findAll { measurement(it, "Nucleus: Area µm^2") < 15 }
    hierarchy.removeObjects(toDelete, true)
    fireHierarchyUpdate(hierarchy)
    */

    def detections = hierarchy.getDetectionObjects()

    /* print("Number of  detections after deleting small nuclei: " + detections.size())
    fireHierarchyUpdate(hierarchy)
    */

    detections.each { it.setPathClass(it.getParent().getPathClass()) }

    // This is necessary to update the image hierarchy.
    fireHierarchyUpdate(hierarchy)

    image.saveImageData(imageData)
}

def export(image, group, outputFiletype) {
    // Method to export measurements

    // Separate each measurement value in the output file with a comma (.csv file)
    String separator = ","

    for (datatype in ["detections", "annotations"]) {
        if (datatype == "detections") {
            print("Exporting detections...")
            exportType = PathDetectionObject.class
        } else if (datatype == "annotations") {
            print("Exporting annotations...")
            exportType = PathAnnotationObject.class
        } else {
            throw new Exception("Datatype not found.")
        }

        // Choose your *full* output path
        outputDir = new File(buildFilePath(OUT_ROOT, OUT_DIR, group))
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }
        outputPath = buildFilePath(OUT_ROOT, OUT_DIR, group, image.getImageName() + "_" + datatype)

        if (outputFiletype == "json" || outputFiletype == "both") {
            boolean prettyPrint = false
            // false results in smaller file sizes and thus faster loading times, at the cost of nice formating
            hierarchy = image.readImageData().getHierarchy()

            if (datatype == "detections") {
                predicate = { it.isDetection() }
            } else if (datatype == "annotations") {
                predicate = { it.isAnnotation() }
            }

            objectsToExport = getObjects(hierarchy=hierarchy, predicate=predicate)

            gson = GsonTools.getInstance(prettyPrint)

            outputFile = new File(outputPath + ".json")
            outputFile.withWriter("UTF-8") {
                gson.toJson(objectsToExport, it)
            }
        }
        if (outputFiletype == "csv" || outputFiletype == "both") {
            outputFile = new File(outputPath + ".csv")

            // Create the measurementExporter and start the export
            new MeasurementExporter()
            .imageList([image])            // Images from which measurements will be exported
            .separator(separator)                 // Character that separates values
            //.includeOnlyColumns(columnsToInclude) // Columns are case-sensitive
            .exportType(exportType)               // Type of objects to export
            .exportMeasurements(outputFile)        // Start the export process
        }
    }
    print "Export Done!"
}
