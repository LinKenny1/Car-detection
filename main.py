import detector
import draw_boxes
from PIL import Image
import matplotlib.pyplot as plt

def run():
	imagePath = 'Test Images\\test7.png'
	bboxes, scores = detector.testImage(imagePath, applyNMS=True)
	print(bboxes)

	img = Image.open(imagePath)
	img = draw_boxes.drawResultsOnImage(img, bboxes, scores)

	plt.imshow(img)
	plt.show()

run()