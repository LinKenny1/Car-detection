import numpy as np
import configuration as cfg
from skimage.feature import hog

def extractHOGfeatures(img):

	fd, hog_img = hog(img,
				      orientations= cfg.HOG_orientations,
					  pixels_per_cell = cfg.HOG_pixels_per_cell,
					  cells_per_block = cfg.HOG_cells_per_block,
					  block_norm = cfg.HOG_norm, 
					  visualise = True)
	return fd, hog_img
