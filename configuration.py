##############################
# FEATURE EXTRACTION Settings
##############################

# HOG parameters

HOG_orientations = 9
HOG_pixels_per_cell = (8, 8)
HOG_cells_per_block = (2, 2)
HOG_norm = 'L2'


###################
# DATASET Settings
###################

datasetRoot = 'C:\\Users\\Guillermo Herrera\\Documents\\USB\\Electivas\\Computer Vision\\Proyecto Detección de Objetos\\Project Folder\\DataSets\\'
positive_folder = 'Vehicle'
negative_folder = 'Non-Vehicle'
positive_test = 'Test Vehicle'
negative_test = 'Test Non-Vehicle'

positiveInputPath = datasetRoot + positive_folder
negativeInputPath = datasetRoot + negative_folder

positiveTestPath = datasetRoot + positive_test
negativeTestPath = datasetRoot + negative_test

positiveFeaturesPath = 'Features\\HOG\\' + positive_folder
negativeFeaturesPath = 'Features\\HOG\\' + negative_folder

positiveTestFeaturesPath = 'Features\\HOG\\' + positive_test
negativeTestFeaturesPath = 'Features\\HOG\\' + negative_test


#####################
# SVM MODEL Settings
#####################

modelPath = 'Models\\SVM_HOG.model'

SVM_C = 0.001#0.01
SVM_penalty = 'l2'
SVM_dual = False
SVM_tolerance = 0.0001
SVM_fit_intercept = True
SVM_intercept_scaling = 100


################
# TEST settings
################

#Sliding window size
window_shape = (64,64)
window_margin = 16
window_step = 16

#Decision threshold for classification
decision_threshold = 0.6 #0.66

#Downscale factor for the pyramid
downScaleFactor = 1.2#1.1

#Padding added to the test images. Used to detect objects at the borders
padding = 16

#Overlapping threshold
nmsOverlapThreshold = 0.65
