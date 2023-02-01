import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("pennies.jpg")
blurred = cv.medianBlur(img, 25)
gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 160, 255, cv.THRESH_BINARY_INV)

dist = cv.distanceTransform(thresh, cv.DIST_L2, 5)

plt.figure()
plt.subplot(121)
plt.imshow(thresh)
plt.subplot(122)
plt.imshow(dist)
plt.show()