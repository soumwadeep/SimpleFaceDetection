import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

test1 = cv2.imread('Images/1.jpeg')

gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_img, cmap='gray')

faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

print('Faces found: ', len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(test1, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(convertToRGB(test1))

def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    img_copy = np.copy(colored_img)

    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_copy

#2nd Test
test2 = cv2.imread('Images/2.jpeg')

faces_detected_img = detect_faces(haar_face_cascade, test2)

plt.imshow(convertToRGB(faces_detected_img))

test2 = cv2.imread('Images/3.jpeg')

faces_detected_img = detect_faces(haar_face_cascade, test2)

plt.imshow(convertToRGB(faces_detected_img))

test2 = cv2.imread('Images/4.jpeg')

faces_detected_img = detect_faces(haar_face_cascade, test2, scaleFactor=1.2)

plt.imshow(convertToRGB(faces_detected_img))

lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')

test2 = cv2.imread('Images/2.jpeg')

faces_detected_img = detect_faces(lbp_face_cascade, test2)

plt.imshow(convertToRGB(faces_detected_img))

test2 = cv2.imread('Images/3.jpeg')

faces_detected_img = detect_faces(lbp_face_cascade, test2)

plt.imshow(convertToRGB(faces_detected_img))

haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')

test1 = cv2.imread('Images/5.jpeg')

test2 = cv2.imread('Images/6.jpeg')

t1 = time.time()

haar_detected_img = detect_faces(haar_face_cascade, test1)

t2 = time.time()

dt1 = t2 - t1

t1 = time.time()

lbp_detected_img = detect_faces(lbp_face_cascade, test1)

t2 = time.time()

dt2 = t2 - t1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.set_title('Haar Detection time: ' + str(round(dt1, 3)) + ' secs')
ax1.imshow(convertToRGB(haar_detected_img))

ax2.set_title('LBP Detection time: ' + str(round(dt2, 3)) + ' secs')
ax2.imshow(convertToRGB(lbp_detected_img))

t1 = time.time()

haar_detected_img = detect_faces(haar_face_cascade, test2)

t2 = time.time()

dt1 = t2 - t1

t1 = time.time()

lbp_detected_img = detect_faces(lbp_face_cascade, test2)

t2 = time.time()

dt2 = t2 - t1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.set_title('Haar Detection time: ' + str(round(dt1, 3)) + ' secs')
ax1.imshow(convertToRGB(haar_detected_img))

ax2.set_title('LBP Detection time: ' + str(round(dt2, 3)) + ' secs')
ax2.imshow(convertToRGB(lbp_detected_img))
