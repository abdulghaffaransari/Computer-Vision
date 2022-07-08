import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np


# load model
model = load_model("MobileNet_model.h5")


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("1.mp4")
label = {0:"With Mask",1:"Without Mask"}
color_label = {0: (0,255,0),1 : (0,0,255)}
while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])
        print(max_index)
        emotions = ('with_mask', 'without_mask')
        predicted_emotion = emotions[max_index]
        if max_index == 1:
            # cv2.rectangle(test_img, (x, y), (x + w, y + h), color_label[0], 3)
            # cv2.rectangle(test_img, (x, y), (x + w, y), color_label[0], -1)
            # cv2.putText(test_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            plt.imshow(test_img)
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
        elif max_index==0:
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 2,cv2.LINE_AA)
            # cv2.rectangle(test_img, (x, y), (x + w, y + h), color_label[1], 3)
            # cv2.rectangle(test_img, (x, y - 50), (x + w, y), color_label[1], -1)
            # cv2.putText(test_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            plt.imshow(test_img)


    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows