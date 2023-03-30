import cv2
import os
import tensorflow as tf
import pandas as pd
import re
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.title("表情による感情推定　デモ")
st.markdown("画像から顔検出し、抽出画像を表情で5分類します。元画像に分類結果と確信度を追記します。")
st.write('顔検出モデル：haar_cascade_face_detection')
st.write('表情分類モデル：network-5Labels')
st.write('分類ラベル：Surprise, Neutral, Anger, Happy, Sad')
st.write('参考：https://github.com/rondinellimorais/facial-expression-recognition')
st.write('未対応：精度&安定性検証、精度向上、描画の視認性向上')

face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml')

settings = {
	'scaleFactor': 1.3, 
	'minNeighbors': 5, 
	'minSize': (50, 50)
}

labels = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']

model = tf.keras.models.load_model('network-5Labels.h5')

img_file_buffer = st.file_uploader("ファイルを選択", type=["jpg", "jpeg", "png"])

# どちらを選択しても後続の処理は同じ
if img_file_buffer:
    img_file_buffer_2 = Image.open(img_file_buffer)
    img = np.array(img_file_buffer_2)
    det_img = np.array(img_file_buffer_2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = face_detection.detectMultiScale(gray, **settings)

    for x, y, w, h in detected:
        det_img = cv2.rectangle(det_img, (x, y), (x+w, y+h), (0, 0, 0), 2)
        face = gray[y+5:y+h-5, x+20:x+w-20]
        face = cv2.resize(face, (48,48))
        face = face/255.0

        model_pred = model.predict(np.array([face.reshape((48,48,1))]))
        predictions = model_pred.argmax()
        percent = model_pred[0][predictions]
        percent = np.round(percent, decimals=2)

        state = labels[predictions] + ': '+ str(percent)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(det_img,state,(x+10,y+15), font, 0.5, (0,0,0), 2, cv2.LINE_AA)

    st.image(det_img, use_column_width=True)


