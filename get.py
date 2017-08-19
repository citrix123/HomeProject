import numpy as np
import cv2
import time

path = '/home/citrix/Documents/Projects/opencv/data/haarcascades/'
face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')

def initVideoDevice(capDevice):
    return cv2.VideoCapture(1)


def renderCamera(cap):
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edged = cv2.Canny(gray, 30, 200)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
	for (x,y,w,h) in faces:
	    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = frame[y:y+h, x:x+w]
	    showFrame('crop', roi_gray)
        
	showFrame('img',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def showFrame(name , frameObj):
    cv2.imshow(name, frameObj)


def destroyDevice(arg):
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main(device):
    renderCamera(initVideoDevice(0))

if __name__ == '__main__':
    main(0)
