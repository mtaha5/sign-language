import streamlit as st
st.set_page_config(
    page_title="Sign Language app",
    page_icon="✋",
    layout="centered"
)
#NAVEGATION BAR
st.sidebar.title("HandSpeak Dynamics")
pagina_seleccionada = st.sidebar.radio("Select an option", ["About", "Learning Area","Word Prediction"])
#MAIN CONTENT
#PAGE 1
if pagina_seleccionada == "About":
    st.markdown("<h1 style='text-align: center; font-size: 70px;'>HandSpeak Dynamics", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 28px;'><strong>SPEAK WITH YOUR HANDS, HEAR WITH YOUR EYES</strong></p>", unsafe_allow_html=True)

    #LOGO
    file_path_logo = "Logo.PNG"
    width_streamlit = 350
    height_streamlit = 350

    ##Show the logo with desired size
    from PIL import Image
    import io
    from base64 import b64encode
    imagen = Image.open(file_path_logo)
    imagen = imagen.resize((width_streamlit, height_streamlit))
    ##convert the image in to bytes
    imagen_bytes = io.BytesIO()
    imagen.save(imagen_bytes, format='PNG')
    ##Show image on Streamlit
    st.markdown(
        f"<div style='display: flex; justify-content: center;'>"
        f"<img src='data:image/png;base64,{b64encode(imagen_bytes.getvalue()).decode()}' "
        f"style='width: {width_streamlit}px; height: {height_streamlit}px;'>"
        f"</div>",
        unsafe_allow_html=True
    )
    # bold and cetered text
    st.write("<p style='text-align: center;'><strong> ", unsafe_allow_html=True)#Empty line
    st.markdown("<p style='text-align: center; font-size: 20px;'><strong>Welcome to HandSpeak Dynamics, where we embark on a transformative journey to redefine communication through groundbreaking Sign Language Translation. Our aim is to break down barriers, enabling seamless and instant communication between the Deaf and hearing communities, fostering a world where everyone's voice is heard.</strong></p>", unsafe_allow_html=True)
#PAGE 2
elif pagina_seleccionada == "Learning Area":
    st.title("Learning Area")
    import streamlit as st
    import cv2
    import numpy as np
    import mediapipe as mp
    from tensorflow.keras import models
    from PIL import Image

    # Print an image of sign vocabulary
    ruta_imagen = "custome_alphabet.PNG"
    st.image(ruta_imagen, caption='American Sign Language Vocabulary',use_column_width=True)

    model = models.load_model('model_customdataset_200.h5')


    label_dict = {0: 'A', 1: 'B', 2: 'C',3: 'D', 4: 'E', 5: 'F',
                6: 'G',7: 'H', 8: 'I',  9: 'K',10: 'L', 11: 'M',
                12: 'N', 13: 'O',14: 'P', 15: 'Q', 16: 'R', 17: 'S',
                18: 'T', 19: 'U', 20: 'V', 21: 'W',22: 'X', 23: 'Y'}

    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    def detect_hand(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands_model.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                h, w, _ = image.shape
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h

                mp_draw.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmark.landmark:
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

        img_crop = image[y_min_new:y_max_new, x_min_new:x_max_new]
        return img_crop

    def resize(image_pil, width, height):
        im = Image.fromarray(image_pil)
        ratio = min(width / im.width, height / im.height)
        new_size = (round(im.width * ratio), round(im.height * ratio))
        image_resized = im.resize(new_size, Image.LANCZOS)
        background = Image.new('RGB', (width, height), (255, 255, 255))
        paste_location = ((width - new_size[0]) // 2, (height - new_size[1]) // 2)

        background.paste(image_resized, paste_location)

        return background

    def generate_alphabet():
        alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        return alphabet

    # Generate alphabet
    alphabet = generate_alphabet()

    # Create a drop-down list
    selected_letter = st.selectbox("Select a letter:", alphabet)

    # Display the selected letter
    st.write(f"You selected: {selected_letter}")

    ss = st.session_state
    if 'word_image' not in ss:
        ss.word_image = []


    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_crop = detect_hand(cv2_img)
        img_background = resize(img_crop, 200, 200)

        #st.image(img_background, channels='RGB')

        img_arr = np.asarray(img_background)
        img_arr_exp = np.expand_dims(img_arr, axis=0)

        #image_prediction
        prediction = model.predict(img_arr_exp)
        predicted_class_index = np.argmax(prediction)
        letter_predicted = label_dict[predicted_class_index]

        #strealit title
        st.header( "Prediction" )

        st.write(letter_predicted)
        ss.word_image = letter_predicted

        if selected_letter == letter_predicted:
            st.markdown("<p style='color:green; font-size:24px; font-weight:bold;'>Congratulations</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:red; font-size:24px; font-weight:bold;'>Try again</p>", unsafe_allow_html=True)


############################################
##############FEEDBACK CODE#################
############################################
############################################
    feedback_text = ""  # Clear the feedback text una carpeta para almacenar archivos de texto (si no existe)

    import os
    storage_folder  = "/home/mtaha5/code/mtaha5/sign-language/mtaha5_notebooks/HandSpeak/feedback"

    if not os.path.exists(storage_folder):
        os.makedirs(storage_folder)

    # Widget to introduce the feedback
    feedback_text = st.text_area("If the prediction is not correct please leave your Feedback here")
    #img_file_buffer = np.expand_dims(img_crop_proc, axis=0)

    # botton to save the text in a file
    if st.button("Save feedback"):
        # text file unique name creation
        name_filetex = f"Feedback_saved_{len(os.listdir(storage_folder)) + 1}.txt"
        file_path = os.path.join(storage_folder, name_filetex)

        # save the text in a file .txt
        with open(file_path, "w", encoding="utf-8") as archivo:
            archivo.write(feedback_text)

        if img_background is not None:
            # Image unique name creation
            name_file_img = f"Feedback_saved_{len(os.listdir(storage_folder)) + 1}.png"
            file_path_img = os.path.join(storage_folder, name_file_img)  # Reemplaza storage_folder con tu ruta deseada

            # Save the image with OpenCV
            img_background = np.squeeze(img_arr_exp, axis=0)  # Eliminar la dimensión de lote (batch)
            cv2.imwrite(file_path_img, cv2.cvtColor(img_background, cv2.COLOR_RGB2BGR))

            # Success message
            st.success("Thank you for your feedback")
        else:
            st.success("Feedback saved successfully without an image.")

            # Button to clear the feedback text
            if st.button("Clear Feedback Text"):
                feedback_text = ""
#PAGE 3
elif pagina_seleccionada == "Word Prediction":
    st.title("Word Prediction")

    st.write("<p style=''>Streamlit web application that integrates several features, including word prediction using hand gestures captured through a webcam, interaction with a machine learning model for letter recognition, and email functionality to send messages.<strong> ", unsafe_allow_html=True)#Empty line

    #Streamlit web application that integrates several features, including word prediction using hand gestures captured through a webcam, interaction with a machine learning model for letter recognition, and email functionality to send messages.

    from streamlit_webrtc import webrtc_streamer
    import streamlit as st
    import av
    import cv2
    import mediapipe as mp
    from tensorflow.keras import models
    import numpy as np
    import skimage
    from PIL import Image

    model = models.load_model('model_customdataset_200.h5')

    label_dict = {0: 'A', 1: 'B', 2: 'C',3: 'D', 4: 'E', 5: 'F',
                6: 'G',7: 'H', 8: 'I',  9: 'K',10: 'L', 11: 'M',
                12: 'N', 13: 'O',14: 'P', 15: 'Q', 16: 'R', 17: 'S',
                18: 'T', 19: 'U', 20: 'V', 21: 'W',22: 'X', 23: 'Y'}

    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    def crop_hand(image):
        results = hands_model.process(image)
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                h, w, _ = image.shape
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h

                for lm in hand_landmark.landmark:
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

        img_crop = image[y_min_new:y_max_new, x_min_new:x_max_new]

        return img_crop

    def resize(image_pil, width, height):
        im = Image.fromarray(image_pil)
        ratio = min(width / im.width, height / im.height)
        new_size = (round(im.width * ratio), round(im.height * ratio))
        image_resized = im.resize(new_size, Image.LANCZOS)
        background = Image.new('RGB', (width, height), (255, 255, 255))
        paste_location = ((width - new_size[0]) // 2, (height - new_size[1]) // 2)

        background.paste(image_resized, paste_location)

        return background

    ss = st.session_state
    if 'word' not in ss:
        ss.word = []

    class VideoProcessor:
        def __init__(self) -> None:
            self.results = None
            self.frame_count = 0
            self.letter = ''

        def detect_hand(self, image):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.results = hands_model.process(image_rgb)
            if self.results.multi_hand_landmarks:
                for hand_landmark in self.results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(image, self.results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                    #image_preprocessing
                    img_crop = crop_hand(image_rgb)
                    img_background = resize(img_crop, 200, 200)
                    img_arr = np.asarray(img_background)
                    img_arr_exp = np.expand_dims(img_arr, axis=0)

                    # imageSize = 48
                    # img_file = skimage.transform.resize(img_crop, (imageSize, imageSize, 3))
                    # img_arr = np.asarray(img_file)
                    # img_arr_exp = np.expand_dims(img_arr, axis=0)

                    #image_prediction
                    prediction = model.predict(img_arr_exp)
                    predicted_class_index = np.argmax(prediction)
                    letter_predicted = label_dict[predicted_class_index]

                    cv2.putText(image , letter_predicted, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.letter = letter_predicted

            return image

        def recv(self, frame):
            frm = frame.to_ndarray(format="bgr24")
            hand_detected = self.detect_hand(frm)

            return av.VideoFrame.from_ndarray(hand_detected, format='bgr24')

    stream_vid =webrtc_streamer(key="key", video_processor_factory=VideoProcessor)

    record_letter = st.button("type letter")
    if record_letter:
        ss.word.append(stream_vid.video_processor.letter)
    with st.form("word_form"):
        st.markdown("".join(ss.word))
        if st.form_submit_button("Clear Word"):
            ss.word = []  # Clear the word when the button is clicked

    #-----------------------------------------------------------------------------------
    from collections import defaultdict
    from nltk import ngrams
    from nltk.probability import FreqDist
    from sklearn.naive_bayes import MultinomialNB
    import numpy as np

    words = ["Data", "science", "is", "an", "ever-evolving", "field", "that", "fascinates", "us.",
            "At", "SLB,", "we", "embrace", "data", "science", "wholeheartedly.", "We", "love", "data",
            "science", "in", "SLB", "because", "it", "fuels", "innovation,", "drives", "decisions,",
            "and", "propels", "us", "forward.", "Python,", "with", "its", "versatility", "and",
            "simplicity,", "plays", "a", "pivotal", "role", "in", "our", "data", "endeavors.",
            "From", "data", "wrangling", "to", "complex", "modeling,", "Python", "empowers", "us",
            "to", "extract", "insights", "and", "craft", "solutions.", "AI,", "the", "pinnacle", "of",
            "technological", "advancement,", "intertwines", "seamlessly", "with", "our", "data",
            "science", "journey.", "We", "leverage", "AI", "to", "enhance", "predictive", "models,",
            "automate", "processes,", "and", "uncover", "hidden", "patterns.", "In", "SLB,", "the",
            "synergy", "between", "data", "science,", "Python,", "and", "AI", "is", "palpable—it",
            "forms", "the", "bedrock", "of", "our", "quest", "for", "excellence.", "Our", "passion",
            "for", "data", "science", "in", "SLB", "knows", "no", "bounds.", "Python", "and", "AI",
            "stand", "as", "pillars,", "fortifying", "our", "exploration,", "experimentation,", "and",
            "pursuit", "of", "cutting-edge", "solutions.", "We", "believe", "in", "the", "transformative",
            "power", "of", "data", "and", "the", "boundless", "possibilities", "that", "Python", "and",
            "AI", "bring", "to", "the", "table.", "Join", "us", "in", "our", "mission", "to", "harness",
            "the", "potential", "of", "data", "science,", "to", "celebrate", "Python's", "elegance,",
            "and", "to", "delve", "deeper", "into", "the", "realms", "of", "AI.", "Together,", "let's",
            "shape", "the", "future", "of", "technology", "through", "innovation,", "curiosity,", "and",
            "a", "profound", "love", "for", "data.", "We", "are", "SLB,", "where", "data", "science",
            "thrives,", "and", "the", "amalgamation", "of", "Python", "and", "AI", "opens", "doors",
            "to", "new", "horizons.", "Data-science", 'Data-science', 'Data-science', 'You', 'You', 'You',
            'HandSpeak Dynamics', 'from', 'from', 'from']

    words = [word.lower() for word in words]

    # Function to generate character n-grams
    def generate_ngrams(word, n):
        return [''.join(gram) for gram in ngrams(word, n)]

    # Create a dictionary of n-grams and their associated words
    ngram_dict = defaultdict(list)
    for word in words:
        for n in range(1, 6):  # Range of n-grams (3 to 6 in this case)
            ngram_dict[n].extend(generate_ngrams(word, n))

    # Build the training dataset
    X = []  # Features (character n-grams)
    y = []  # Labels (words)
    for n, ngrams_list in ngram_dict.items():
        for ngram in ngrams_list:
            X.append(ngram)
            y.append([word for word, grams in ngram_dict.items() if ngram in grams][0])

    # Convert features to binary vectors using occurrence frequency
    vectorizer = FreqDist(X)
    X_train = []
    for word in words:
        word_ngrams = generate_ngrams(word, 3)  # Generate n-grams for the word
        word_vector = [1 if ngram in word_ngrams else 0 for ngram in vectorizer]
        X_train.append(word_vector)

    # Train a classification model (Naive Bayes in this case)
    clf = MultinomialNB()
    clf.fit(X_train, words)

    # This is the predict_word function you provided
    def predict_word(partial_letters):
        partial_ngrams = generate_ngrams(partial_letters, 3)  # Generate n-grams for partial letters
        partial_vector = [1 if ngram in partial_ngrams else 0 for ngram in vectorizer]
        predicted_words = clf.predict_proba([partial_vector])[0]

        # Filter the predictions to consider words starting with the input letters
        predictions = [
            (word, prob)
            for word, prob in zip(clf.classes_, predicted_words)
            if prob > 0 and word.startswith(partial_letters.lower())
        ]

        # Sort predictions by probability and return the top three with highest probability
        top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:3]
        return top_predictions


    # Define the Streamlit interface
    import streamlit as st

    def main():
        word_from_ss = " ".join(ss.word) if hasattr(ss, 'word') else ""
        selected_word = st.session_state.get('selected_word', '')  # Get the value of the selected word from the session

        text = st.text_input("Text:", value=selected_word if selected_word else word_from_ss)

        words=text.split()
        partial_letters=''

        show_prediction_button = True  # Initialize the show_prediction_button variable

        if len(words) > 0:
            partial_letters = words[-1]

        if partial_letters:
            predictions = predict_word(partial_letters)
            if predictions:
                num_cols = 3  # Number of columns for display
                num_predictions = len(predictions)
                num_rows = -(-num_predictions // num_cols)  # Calculate the number of rows needed

                # Display predictions in columns
                for row in range(num_rows):
                    cols = st.columns(num_cols)
                    for col in range(num_cols):
                        index = row * num_cols + col
                        if index < num_predictions:
                            prediction = predictions[index]
                            if cols[col].button(prediction[0]):
                                if partial_letters and prediction[0].startswith(partial_letters.lower()):
                                    # Replace the partial sequence with the predicted word
                                    ss.word = ss.word[:-len(partial_letters.split())]
                                    ss.word.append(prediction[0])
                                    ss.word.append(" ")  # Add a space after the word
                                else:
                                    ss.word.append(prediction[0])  # Add the predicted word to the session state
                                    ss.word.append(" ")  # Add a space after the word
                                st.session_state['selected_word'] = ""  # Clear the selected word
                                st.experimental_rerun()  # Rerun the app to update the text field
                                break
        else:
            st.write("No words found for the provided partial letters.")

    if __name__ == "__main__":
        main()
############################################
##############EMAIL CODE####################
############################################
############################################
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    import smtplib
    import io

    # get letters signature and picture information you want to send by email
    final_letters = ss.word
    signature = "BRegards<br>HandSpeak Dynamics" # Signature
    logoemail = "logoemail.png"

    # Create the object MIMEMultipart to send the email
    mensaje = MIMEMultipart()
    mensaje['From'] = 'handspeakdynamics@gmail.com'  # Reemplaza con tu dirección de correo
    mensaje['To'] = ", ".join(['eng.mazinzain@gmail.com', 'icruz8@slb.com', 'tahaosman242@gmail.com'])  # Reemplaza con la dirección de correo del destinatario
    mensaje['Subject'] = 'Sign Languaje Translator'

    # Personalized message
    #personalized_msg = "Hello\n\nI've created a personalized message for you using the HandSpeak Dynamics application:"
    personalized_msg = "<p>Hello,</p>\n<p>I've created a personalized message for you using the HandSpeak Dynamics application.</p>"
    # Main body of the email
    body_msg = '<strong>' + ' '.join(ss.word) + '</strong>'  # Assuming ss.word is a list of words
    body_msg += f'<p style="margin-bottom: 0;"><br>{signature}</p>'
    body_msg += '<p style="margin-top: 0;"><img src="cid:attachedimage" alt="attachedimage"></p>'

    mensaje.attach(MIMEText(personalized_msg + '<br><br>' + body_msg, 'html'))


    # Adjuntar la imagen al mensaje con un identificador (CID)
    with open(logoemail, 'rb') as file:
        attachedimage = MIMEImage(file.read(), name="attachedimage.jpg")
        attachedimage.add_header('Content-ID', '<attachedimage>')
        mensaje.attach(attachedimage)

        # Conect to server SMTP
    smtp_server = 'smtp.gmail.com'  # Email server SMTP
    smtp_port = 587  # Port of server SMTP
    user = 'handspeakdynamics@gmail.com'  # Your email
    password = 'wjar howm mvuq crcs'  # your password gmal apps

    #password = '@Zxcvbnm'  # your password gmal

    if st.button('Send by email'):
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(user, password)

            # Concatenate the message and sent by email
            mail_text = mensaje.as_string()
            server.sendmail(user, ['eng.mazinzain@gmail.com', 'icruz8@slb.com', 'tahaosman242@gmail.com'] , mail_text)

        # Succes message
        st.success('Email successfully sent')
