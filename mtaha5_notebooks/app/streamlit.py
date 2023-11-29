import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

mp_hand = mp.solutions.hands
hands = mp_hand.Hands()

mp_drawing_utils = mp.solutions.drawing_utils

img_file_buffer = st.camera_input("test")

if img_file_buffer:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    results = hands.process(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    h, w, c = cv2_img.shape
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h

    hand_landmarks = results.multi_hand_landmarks

    #print(results.multi_hand_landmarks)

    # if hand_landmarks:
    #     for handLMs in hand_landmarks:
    #         mp_drawing_utils.draw_landmarks(cv2_img, handLMs, mp_hand.HAND_CONNECTIONS)

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h

            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

    extra = 30

    if x_max+extra > w:
        x_max_new = w
    else:
        x_max_new = x_max+extra
    if x_min-extra < 0:
        x_min_new = 0
    else:
        x_min_new = x_min-extra
    if y_max+extra > h:
        y_max_new = h
    else:
        y_max_new = y_max+extra
    if y_min-extra < 0:
        y_min_new = 0
    else:
        y_min_new = y_min-extra

    #asp1 = max(y_max_new - y_min_new, x_max_new - x_min_new)
    #img_crop = cv2_img[y_min_new:(y_min_new+asp1), x_min_new:(x_min_new+asp1)]
    #cv2.rectangle(cv2_img, (x_min_new, y_min_new), (x_max_new, y_max_new), (0, 255, 0), 2)
    img_crop = cv2_img[y_min_new:y_max_new, x_min_new:x_max_new]

    st.markdown('x_min = {}'.format(x_min))
    st.markdown('x_max = {}'.format(x_max))
    st.markdown('y_min = {}'.format(y_min))
    st.markdown('y_max = {}'.format(y_max))
    st.markdown(cv2_img.shape)


    with open('image_test.npy', 'wb') as f:
        np.save(f ,img_crop)
