import numpy as np
import cv2
import time
import dlib

path = '/home/citrix/Documents/Projects/opencv/data/haarcascades/'
face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("res/shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the down
count = 0


def initVideoDevice(capDevice):
    return cv2.VideoCapture(1)


def renderCamera(cap):
    name = ""
    train = 0
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
	    roi_img = frame[y:y+h, x:x+w] #RGB image
	    roi_color = frame[y:y+h, x:x+w]
	    showFrame('crop', roi_img)
	    if train == 1:
		saveImages(roi_img, name)
	
	    getFacialCircles(roi_gray, frame)

	showFrame('img',frame)
	
	k = cv2.waitKey(1)
	print k
        if k & 0xFF == ord('q'):
            break
	elif k & 0xFF == ord('t'):
	    print "Pressed T"
	    name = raw_input("What is your Name?")
	    train = 1
	elif k & 0xFF == ord('s'):
	    train = 0
	    print "Taining Stopped"
	

def getFacialCircles(imgObj, frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(imgObj)
    detections = detector(clahe_image, 1)
    
    for k,d in enumerate(detections): #For each detected face    
	shape = predictor(clahe_image, d) #Get coordinates
	for i in range(1,68): #There are 68 landmark points on each face
	    cv2.circle(imgObj, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
    showFrame("FacialCircles", imgObj)


def showFrame(name , frameObj):
    cv2.imshow(name, frameObj)


def destroyDevice(arg):
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def saveImages(imgObj, Name):
    global count
    count = count+1
    path = "./Images/"
    Name = path + "/" + Name + "_" + str(count) + ".jpg"
    print Name
    cv2.imwrite(Name, imgObj)


def main(device):
    renderCamera(initVideoDevice(0))

if __name__ == '__main__':
    main(1)
