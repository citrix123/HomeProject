import numpy as np
import cv2
import time
cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # ret,thresh = cv2.threshold(blurred,127,255,cv2.THRESH_TOZERO_INV)

    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    # ret,thresh = cv2.threshold(thresh, 60, 255, cv2.THRESH_BINARY)
    # Display the resulting frame
    cv2.imshow('frame',th2)
    # cv2.imshow('blurred',blurred)
    # laplacian = cv2.Laplacian(thresh,cv2.CV_64F)
    # cv2.imshow('frame2',blurred)
    # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    # screenCnt = None
    # for c in cnts:
    #     peri = cv2.arcLength(c , True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # 	# if our approximated contour has four points, then
    # 	# we can assume that we have found our screen
    # 	if len(approx) > 11:
    # 		screenCnt = approx
    # 		break
    #
    # # mask = np.zeros_like(frame)
    # cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 3)
    # print screenCnt
    # out = np.zeros_like(frame)
    # out[mask == 255] = frame[mask == 255]
    # cv2.imshow("Game Boy Screen", frame)
    # cv2.imshow("output", out)
    # im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
