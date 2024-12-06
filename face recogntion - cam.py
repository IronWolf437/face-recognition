import cv2
import numpy as np
import face_recognition as fr
import os

path = "D:/Courses/Programming Courses/Artificial Intelligence/AMIT-AI/AI_code/5. computer vision/Session 1 - Image Processing/data"
images = []
classNames = []


# show images in file:
myList = os.listdir(path)
print(myList)


# for read images and export names:
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# encode the images in file: 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('number of images:',len(encodeListKnown))
print('Encoding Complete...')


# get image from camera:
cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    img= cv2.flip(img, 1)
    imgS = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = fr.face_locations(imgS)
    encodesCurFrame = fr.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*5, x2*5, y2*5, x1*5
        if matches[matchIndex] and faceDis[matchIndex] < 0.6:
            name = classNames[matchIndex].upper()
            print(name)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,225,0),2)
            cv2.rectangle(img,(x1,y2-25),(x2,y2),(0,225,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255),2)
        else:
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-25),(x2,y2),(0,0,255), cv2.FILLED)
            cv2.putText(img, 'unknown', (x1+6,y2-6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255),2)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
