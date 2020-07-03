import configuration as cfg

import numpy as np
import pickle
import os
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def loadImageFeatures(featuresPath):
	print('Loading features from' + str(featuresPath))
	file = open(featuresPath , 'rb')
	return pickle.load(file)

def run():

	#List all the files .feat in the positive directory
	positiveList = os.listdir(cfg.positiveFeaturesPath)
	#Select only files that end in .feat
	positiveList = list(filter(lambda element: '.feat' in element, positiveList))

	#List all the files .feat in the negative directory
	negativeList = os.listdir(cfg.negativeFeaturesPath)
	#Select only files that end in .feat
	negativeList = list(filter(lambda element: '.feat' in element, negativeList))

	#List all the files .feat in the positive test directory
	positiveTestList = os.listdir(cfg.positiveTestFeaturesPath)
	#Select only files that end in .feat
	positiveTestList = list(filter(lambda element: '.feat' in element, positiveTestList))

	#List all the files .feat in the negative test directory
	negativeTestList = os.listdir(cfg.negativeTestFeaturesPath)
	#Select only files that end in .feat
	negativeTestList = list(filter(lambda element: '.feat' in element, negativeTestList))

	#Count the samples
	positiveSamplesCount = len(positiveList)
	negativeSamplesCount = len(negativeList)
	samplesCount = positiveSamplesCount + negativeSamplesCount

	positiveTestSamplesCount = len(positiveTestList)
	negativeTestSamplesCount = len(negativeTestList)
	samplesTestCount = positiveTestSamplesCount + negativeTestSamplesCount


	#Load the features of the first element to obtain the size of the feature vector
	filepath = cfg.positiveFeaturesPath + '\\' + positiveList[0]
	file = open(filepath, 'rb')
	features = pickle.load(file)
	featuresLength = len(features)

	#Initialize the structure that will be passed to the model for training
	X = np.zeros(shape=(samplesCount, featuresLength))
	aux_positive = np.ones(shape=(1,positiveSamplesCount))
	aux_negative = np.ones(shape=(1,negativeSamplesCount))
	Y = np.append(aux_positive,(-1*aux_negative))

	X_test = np.zeros(shape=(samplesTestCount, featuresLength))
	aux_test_positive = np.ones(shape=(1,positiveTestSamplesCount))
	aux_test_negative = np.ones(shape=(1,negativeTestSamplesCount))
	Y_test = np.append(aux_test_positive,(-1*aux_test_negative))

	#Load all the positive features vectors to X
	count = 0
	for filename in positiveList:
		filepath = cfg.positiveFeaturesPath + '\\' + filename
		X[count] = loadImageFeatures(filepath)
		count += 1

	#Load all the negative features vectors to X
	for filename in negativeList:
		filepath = cfg.negativeFeaturesPath + '\\' + filename
		X[count] = loadImageFeatures(filepath)
		count += 1

	#Load all the positive features vectors to X_test
	count = 0
	for filename in positiveTestList:
		filepath = cfg.positiveTestFeaturesPath + '\\' + filename
		X_test[count] = loadImageFeatures(filepath)
		count += 1

	#Load all the negative features vectors to Y_test
	for filename in negativeTestList:
		filepath = cfg.negativeTestFeaturesPath + '\\' + filename
		X_test[count] = loadImageFeatures(filepath)
		count += 1

	print('Training SVM...')
	model = svm.LinearSVC(penalty=cfg.SVM_penalty,
	                      dual=cfg.SVM_dual,
	                      tol=cfg.SVM_tolerance,
	                      C=cfg.SVM_C,
	                      fit_intercept=cfg.SVM_fit_intercept,
	                      intercept_scaling=cfg.SVM_intercept_scaling)

	model.fit(X,Y)

	print('Model score')
	print(model.score(X,Y))

	print('Model test score')
	print(model.score(X_test, Y_test))

	modelDirectory, modelFilename = os.path.split(cfg.modelPath)
	if not os.path.exists(modelDirectory):
		os.makedirs(modelDirectory)

	outputModelFile = open(cfg.modelPath, 'wb')
	pickle.dump(model, outputModelFile)

run()
