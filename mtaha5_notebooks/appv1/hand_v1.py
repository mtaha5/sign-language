import av
import cv2
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from datetime import datetime


mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

def live_hand_detection(play_state):

    class HandProcessor(VideoProcessorBase):

        def __init__(self) -> None:
            self.results = None
            self.frame_count = 0

        def detect_hand(self, image):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.results = hands_model.process(image_rgb)  # Usar el modelo de manos inicializado
            if self.results.multi_hand_landmarks:
                for hand_landmark in self.results.multi_hand_landmarks:
                    h, w, _ = image.shape
                    mp_draw.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)
                    # landmarks = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]
                    # cv2.rectangle(image, (min(landmarks)[0], min(landmarks, key=lambda x: x[1])[1]),
                    #               (max(landmarks)[0], max(landmarks, key=lambda x: x[1])[1]),
                    #               (0, 255, 0), 2)
            return image

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            hand_detected_image = self.detect_hand(image)

            if self.frame_count % 5 == 0:
                image_path = f"captured_frame_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                cv2.imwrite(image_path, frame.to_ndarray())
                st.success(f"Frame saved as: {image_path}")
            self.frame_count += 1

            return av.VideoFrame.from_ndarray(hand_detected_image, format="bgr24")



    stream = webrtc_streamer(
        key="hand-detection",
        mode=WebRtcMode.SENDRECV,
        desired_playing_state=play_state,
        video_processor_factory=HandProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

play_state = True
live_hand_detection(play_state)

#cap = cv2.VideoCapture(0)

# webrtc_ctx = webrtc_streamer(
#     key="video-sendonly",
#     mode=WebRtcMode.SENDONLY,
#     media_stream_constraints={"video": True},
# )

# while True:
#     if webrtc_ctx.video_receiver:
#         video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
#         img_rgb = video_frame.to_ndarray(format="rgb24")

#         ret, img = video_frame.read()
#         h, w, c = img.shape
#         results = hands.process(img)

#         if results.multi_hand_landmarks:
#             for hand_landmark in results.multi_hand_landmarks:
#                 mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
