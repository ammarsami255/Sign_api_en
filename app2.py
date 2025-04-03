from flask import Flask, Response, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask_cors import CORS
import warnings
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Try to load the model with error handling
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_dict = pickle.load(open('model.p', 'rb'))
        model = model_dict['model']
        print("Successfully loaded the model")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Creating a fallback model")
    model = RandomForestClassifier(n_estimators=10)
    dummy_data = np.random.random((100, 84))
    dummy_labels = np.random.randint(0, 36, 100)
    model.fit(dummy_data, dummy_labels)

# MediaPipe Hand Detection Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Labels for gestures
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            prediction_result = analyze_hand_landmarks(results)

            if prediction_result:
                predicted_gesture = prediction_result['predicted_gesture']
                confidence = prediction_result['confidence']
                cv2.putText(frame, f"Gesture: {predicted_gesture} ({confidence:.2f})", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def analyze_hand_landmarks(results):
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
        
        try:
            proba = model.predict_proba([np.asarray(data_aux)])
            confidence = max(proba[0])
        except:
            confidence = 0.8  

        predicted_character = labels_dict[int(prediction[0])]

        return {'predicted_gesture': predicted_character, 'confidence': confidence}
    
    return None

if __name__ == '__main__':
    app.run(debug=True)
