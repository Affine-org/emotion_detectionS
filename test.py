import numpy as np
import cv2
import tensorflow as tf
import streamlit as st

import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from PIL import Image

face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml')

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
settings = {
	'scaleFactor': 1.3, 
	'minNeighbors': 5, 
	'minSize': (50, 50)
}

labels = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']

model = tf.keras.models.load_model('network-5Labels.h5')
st.title("Face emotion app")

# while True:
# ret, img = camera.read()
img_file_buffer = st.file_uploader("ファイルを選択")

# どちらを選択しても後続の処理は同じ
if img_file_buffer:
    img_file_buffer_2 = Image.open(img_file_buffer)
    img = np.array(img_file_buffer_2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = face_detection.detectMultiScale(gray, **settings)

    for x, y, w, h in detected:
        cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
        cv2.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
        face = gray[y+5:y+h-5, x+20:x+w-20]
        face = cv2.resize(face, (48,48))
        face = face/255.0

        predictions = model.predict(np.array([face.reshape((48,48,1))])).argmax()
        state = labels[predictions]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,state,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
        st.markdown("#### あなたの表情は")
        st.markdown("### {}です".format(state))

# cv2.imshow('Facial Expression', img)

# if cv2.waitKey(5) != -1:
#     break

# camera.release()
# cv2.destroyAllWindows()
