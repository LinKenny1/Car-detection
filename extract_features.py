import feature_extractor
import configuration as cfg

import os
import pickle
from skimage import io, util


def run():

	#Create necessary directories to save the features of the images

	#Training set
	if not os.path.exists(cfg.positiveFeaturesPath):
		os.makedirs(cfg.positiveFeaturesPath)
	if not os.path.exists(cfg.negativeFeaturesPath):
		os.makedirs(cfg.negativeFeaturesPath)

	#Test set
	if not os.path.exists(cfg.positiveTestFeaturesPath):
		os.makedirs(cfg.positiveTestFeaturesPath)
	if not os.path.exists(cfg.negativeTestFeaturesPath):
		os.makedirs(cfg.negativeTestFeaturesPath)

	#Extract features for positive samples
	print('Extracting features from images in ' + cfg.positiveInputPath)
	extractAndStoreFeatures(cfg.positiveInputPath, cfg.positiveFeaturesPath)

	#Extract features for negative samples
	print('Extracting features from images in ' + cfg.negativeInputPath)
	extractAndStoreFeatures(cfg.negativeInputPath, cfg.negativeFeaturesPath)

	#Extract features for positive test samples
	print('Extracting features from images in ' + cfg.positiveTestPath)
	extractAndStoreFeatures(cfg.positiveTestPath, cfg.positiveTestFeaturesPath)

	#Extract features for negative test samples
	print('Extracting features from images in ' + cfg.negativeTestPath)
	extractAndStoreFeatures(cfg.negativeTestPath, cfg.negativeTestFeaturesPath)


def extractAndStoreFeatures(inputFolder, outputFolder):

	#List all files
	fileList = os.listdir(inputFolder)
	#Select only files that end with .png
	imagesList = list(filter(lambda element: '.png' in element, fileList))

	for filename in imagesList:
		imagepath = inputFolder + '\\' + filename
		outputpath = outputFolder + '\\' + filename + '.feat'

		if os.path.exists(outputpath):
			print('Features for ' + imagepath + '. Delete the file if you want to replace it.')
			continue

		print('Extracting features for ' + imagepath)

		image = io.imread(imagepath, as_grey=True)
		#Read the image as bytes (pixels with 0-255 values)
		image = util.img_as_ubyte(image)

		#Extract the features
		features, HOG_img = feature_extractor.extractHOGfeatures(image)

		#Savethe features to a file
		outputFile = open(outputpath, 'wb')
		pickle.dump(features, outputFile)
		outputFile.close()

run()

