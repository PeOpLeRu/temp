# SIFT и слежение за выделенным объектом
# Применить алгоритм SIFT ля слежения за объектом с камеры. Выбор объекта реализовать при помощи selectROI.

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def edit_img(img, obj_img):
    orb = cv.SIFT_create()

    key_points1, descriptors1 = orb.detectAndCompute(obj_img, None)
    key_points2, descriptors2 = orb.detectAndCompute(img, None)

    mathcer = cv.BFMatcher()

    matches = mathcer.knnMatch(descriptors1, descriptors2, k=2)

    best = []

    for im1, im2 in matches:
        if im1.distance < 0.75 * im2.distance:
            best.append([im1])

    # print(f"All mathces -> {len(matches)}")
    # print(f"Best matches - > {len(best)}")

    if len(best) > 30:
        src_pts = np.float32([key_points1[m[0].queryIdx].pt for m in best]).reshape(-1, 1, 2)
        dst_pts = np.float32([key_points2[m[0].trainIdx].pt for m in best]).reshape(-1, 1, 2)
        M, hmask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        result = cv.polylines(img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        print("Yes matches")
    else:
        result = img
        print("Not enough matches")
        mask = None

    matches_image = cv.drawMatchesKnn(obj_img, key_points1, img, key_points2, best, None)

    return result

# main code

cv.namedWindow("Camera", cv.WINDOW_GUI_NORMAL)

cam = cv.VideoCapture(0)

roi = None

while cam.isOpened():
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if roi is not None:
        frame = edit_img(gray, roi)
    else:
        x, y, w, h = cv.selectROI("Selection", gray)
        roi = gray[np.int32(y) : np.int32(y + h), np.int32(x) : np.int32(x + w)]
        cv.imshow("ROI", roi)
        cv.destroyWindow("Selection")
        cv.destroyWindow("ROI")

    cv.imshow("Camera", frame)

    key = cv.waitKey(1)
    if key == ord('q') or key == ord('й'):
        break

cam.release()
cv.destroyAllWindows()