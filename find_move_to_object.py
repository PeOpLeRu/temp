import cv2 as cv

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
cam.set(cv.CAP_PROP_EXPOSURE, -4)

cv.namedWindow("Camera", cv.WINDOW_GUI_NORMAL)

roi = None

background = None
backgrounds = []

while cam.isOpened():
    ret, frame = cam.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # gray = cv.GaussianBlur(gray, (21, 21), 0)

    if roi is not None:
        res = cv.matchTemplate(gray, roi, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + roi.shape[1], top_left[1] + roi.shape[0])
        cv.rectangle(frame, top_left, bottom_right, 255, 2)
        cv.imshow("Matching", res)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('1'):
        x, y, w, h = cv.selectROI("Selection", gray)
        roi = gray[int(y) : int(y+h), int(x) : int(x+w)]
        cv.imshow("ROI", roi)
        cv.destroyWindow("Selection")

    cv.imshow("Camera", frame)
    
# -----------------------

cam.release()
cv.destroyAllWindows()