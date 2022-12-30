#公众号：OpenCV与AI深度学习
import cv2
import numpy as np
import imutils
import pytesseract

image = cv2.imread('0.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Otsu", thresh)

dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
dist = (dist * 255).astype("uint8")
cv2.imshow("Dist", dist)

# threshold the distance transform using Otsu's method
_,dist = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Dist Otsu", dist)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)
cv2.imshow("Opening", opening)

black_img = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)

cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
chars = []
# loop over the contours<font></font>
for c in cnts:
  # compute the bounding box of the contour
  (x, y, w, h) = cv2.boundingRect(c)
  if w >= 35 and h >= 100:
    chars.append(c)
    
cv2.drawContours(black_img,chars,-1,(0,255,0),2)
cv2.imshow("chars", black_img)

chars = np.vstack([chars[i] for i in range(0, len(chars))]) 
hull = cv2.convexHull(chars)

# allocate memory for the convex hull mask, draw the convex hull on<font></font>
# the image, and then enlarge it via a dilation<font></font>
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.drawContours(mask, [hull], -1, 255, -1)
mask = cv2.dilate(mask, None, iterations=2)
cv2.imshow("Mask", mask) 

# take the bitwise of the opening image and the mask to reveal *just*<font></font>
# the characters in the image<font></font>
final = cv2.bitwise_and(opening, opening, mask=mask)
cv2.imshow("final", final)

text = pytesseract.image_to_string(final)
# 打印识别后的文本
print(text)

cv2.waitKey()
cv2.destroyAllWindows()