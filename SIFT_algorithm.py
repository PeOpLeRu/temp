# Ключевые точки, алгоритм SIFT

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

single = cv.imread("reeses_puffs.png", 0)
many = cv.imread("many_cereals.jpg", 0)

orb = cv.SIFT_create()

key_points1, descriptors1 = orb.detectAndCompute(single, None)
key_points2, descriptors2 = orb.detectAndCompute(many, None)

mathcer = cv.BFMatcher()

matches = mathcer.knnMatch(descriptors1, descriptors2, k=2)

best = []

for im1, im2 in matches:
    if im1.distance < 0.75 * im2.distance:
        best.append([im1])

print(f"All mathces -> {len(matches)}")
print(f"Best matches - > {len(best)}")

if len(best) > 30:
    src_pts = np.float32([key_points1[m[0].queryIdx].pt for m in best]).reshape(-1, 1, 2)
    dst_pts = np.float32([key_points2[m[0].trainIdx].pt for m in best]).reshape(-1, 1, 2)
    M, hmask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    h, w = single.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    result = cv.polylines(many, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    print("Not enough matches")
    mask = None

matches_image = cv.drawMatchesKnn(single, key_points1, many, key_points2, best, None)

plt.figure()
plt.imshow(matches_image)
plt.show()