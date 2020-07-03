import os
from PIL import Image, ImageDraw, ImageFont, ImageChops

def drawResultsOnImage(img, boxes, scores, bboxColorRGB =(0, 0, 255)):

    boxcount = 0

    draw = ImageDraw.Draw(img)
    textSize = 20
    #fnt = ImageFont.truetype(os.getcwd() + '\\fonts\\FreeMonoBold.ttf', textSize)

    #Draw a bounding box with color bbocColorBGR for every detection in bboxes
    #Draw also the score of that bounding box in it's top-left corner
    for bbox in boxes:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        # Remark the object with a rectangle, and draw the text
        draw.line((x1, y1, x2, y1), fill=bboxColorRGB, width=2)
        draw.line((x2, y1, x2, y2), fill=bboxColorRGB, width=2)
        draw.line((x2, y2, x1, y2), fill=bboxColorRGB, width=2)
        draw.line((x1, y2, x1, y1), fill=bboxColorRGB, width=2)

        #Obtain the score and take it's 4 first decimals
        if scores is not None:
            score = scores[boxcount]
            scoreText = "{0:.4f}".format(score)
            draw.text((x1, y1), scoreText, fill=(255, 255, 255, 255))#, font=fnt)
            boxcount += 1

    return img