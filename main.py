import numpy as np
import argparse
import imutils
from imutils.contours import sort_contours
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# load the input image from disk, convert it to grayscale, and blur,to reduce noise
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)  # increases breaking
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
chars = []

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    if (w >= 15 and w <= 150) and (h >= 20 and h <= 120):
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(
            roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape
        try:

            if tW > tH:
                thresh = imutils.resize(thresh, width=32)

            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)
           
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            chars.append((padded, (x, y, w, h)))
        except:
            print(thresh.shape)
            cv2.imshow("Error", thresh)
            cv2.waitKey(0)


boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
counts = [i for i in range(len(boxes))]
for (pred, (x, y, w, h)) in zip(counts, boxes):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
