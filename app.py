from flask import Flask, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# MediaPipe Hand Detection Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    # Convert the incoming image to an OpenCV-compatible format
    img = Image.open(BytesIO(file.read()))
    img = np.array(img)
    
    # Convert image to RGB
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)
    
    data_aux = []
    x_ = []
    y_ = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            temp_data = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                temp_data.append(x - min(x_))
                temp_data.append(y - min(y_))

            if len(temp_data) == 42:
                data_aux.extend(temp_data)

        while len(data_aux) < 84:
            data_aux.append(0)

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        return jsonify({'predicted_gesture': predicted_character})
    
    return jsonify({'error': 'No hand detected'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
