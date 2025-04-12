from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
from tensorflow import keras
import copy
import itertools
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
port = int(os.getenv('PORT', 5000))

# Load the saved model from file
model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet =  ['1','2','3','4','5','6','7','8','9']
alphabet += list(string.ascii_uppercase)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    return temp_landmark_list

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                df = pd.DataFrame(pre_processed_landmark_list).transpose()
                predictions = model.predict(df, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
                label = alphabet[predicted_classes[0]]
                return jsonify({'label': label}), 200
    return jsonify({'label': 'No hand detected'})

if __name__ == '__main__':
    print("Running Flask app...")
    app.run(debug=True, port=5001)