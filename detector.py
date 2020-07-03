import configuration as cfg
import feature_extractor

import os
import pickle
import math
import nms
import numpy as np
import skimage.util as util
from sklearn import svm
from skimage import io
from skimage.util import pad
from skimage.transform import pyramid_gaussian
from skimage.util.shape import view_as_windows

def testImage(imagePath, decisionThreshold=cfg.decision_threshold, applyNMS=True):

	file = open(cfg.modelPath, 'rb')
	svc = pickle.load(file)

	image = io.imread(imagePath, as_grey=True)
	image = util.img_as_ubyte(image)

	rows, cols = image.shape
	pyramid = tuple(pyramid_gaussian(image, downscale=cfg.downScaleFactor))

	scale = 0
	boxes = None
	scores = None

	for p in pyramid[0:]: 

		if cfg.padding > 0:
			p = pad(p,cfg.padding,'reflect')

		try:
			views = view_as_windows(p, cfg.window_shape, step=cfg.window_step)
		except ValueError:
			break

		num_rows, num_cols, width, height = views.shape
		for row in range(0,num_rows):
			for col in range(0,num_cols):

				auxImage = views[row, col]

				features, hog_img = feature_extractor.extractHOGfeatures(auxImage)

				decision_func = svc.decision_function(features)
				print(decision_func)

				if decision_func > decisionThreshold:
					h, w = cfg.window_shape
					scaleMult = math.pow(cfg.downScaleFactor, scale)

					x1 = int(scaleMult * (col*cfg.window_step - cfg.padding + cfg.window_margin))
					y1 = int(scaleMult * (row*cfg.window_step - cfg.padding + cfg.window_margin))
					x2 = int(x1 + scaleMult*(w - 2*cfg.window_margin))
					y2 = int(y1 + scaleMult*(h - 2*cfg.window_margin))

					bbox = (x1, y1, x2, y2)
					score = decision_func[0]

					if boxes is not None:
						boxes = np.vstack((bbox, boxes))
						scores = np.hstack((score, scores))
					else:
						boxes = np.array([bbox])
						scores = np.array([score])
		scale += 1

	if applyNMS:
		boxes, scores = nms.non_max_suppression_fast(boxes, scores, cfg.nmsOverlapThreshold)

	return boxes, scores
