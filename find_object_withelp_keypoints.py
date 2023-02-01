# Ключевые точки, алгоритм orb

import cv2 as cv
import matplotlib.pyplot as plt

single = cv.imread("reeses_puffs.png")
many = cv.imread("many_cereals.jpg")

orb = cv.ORB_create()

key_points1, descriptors1 = orb.detectAndCompute(single, None)
key_points2, descriptors2 = orb.detectAndCompute(many, None)

print(key_points2)
print(descriptors1)

mathcer = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = mathcer.match(descriptors1, descriptors2)
print(matches[0].distance)

matches = sorted(matches, key=lambda x : x.distance)

matches_image = cv.drawMatches(single, key_points1, many, key_points2, matches[:20], None)

plt.figure()

plt.subplot(121)
plt.imshow(single)

plt.subplot(122)
plt.imshow(many)

plt.figure()
plt.imshow(matches_image)

plt.show()