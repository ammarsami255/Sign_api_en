from flask import Flask, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import tempfile
import os
from collections import Counter

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# MediaPipe Hand Detection Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Static hands for images
hands_static = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# Non-static hands for videos (better tracking)
hands_video = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    # Check file extension to determine if it's an image or video
    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower() if filename else ''
    
    # Process based on file type
    if file_ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv']:
        # Handle video file
        return process_video(file)
    else:
        # Handle image file
        try:
            img = Image.open(BytesIO(file.read()))
            img = np.array(img)
            return process_image(img)
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})

def process_image(img):
    # Convert image to RGB
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands_static.process(frame_rgb)
    
    return analyze_hand_landmarks(results)

def process_video(file):
    # Save the video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    file.save(temp_file.name)
    temp_file.close()
    
    # Open the video file
    cap = cv2.VideoCapture(temp_file.name)
    
    if not cap.isOpened():
        os.unlink(temp_file.name)
        return jsonify({'error': 'Could not open video file'})
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        os.unlink(temp_file.name)
        return jsonify({'error': 'Video file has no frames'})
    
    # Process frames at regular intervals - more frames for better accuracy
    frame_interval = max(1, int(fps / 5))  # Process 5 frames per second
    predictions = []
    timestamps = []
    confidence_scores = []
    
    # Use the video-optimized hand detector
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Process the frame
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                # Get prediction for this frame
                prediction_result = analyze_hand_landmarks_for_video(results)
                
                if 'predicted_gesture' in prediction_result:
                    predictions.append(prediction_result['predicted_gesture'])
                    timestamps.append(frame_idx / fps)  # Convert frame index to seconds
                    confidence_scores.append(prediction_result.get('confidence', 0.5))
            except Exception as e:
                continue  # Skip frames that cause errors
    
    # Release the video capture object and delete temp file
    cap.release()
    os.unlink(temp_file.name)
    
    if not predictions:
        return jsonify({'error': 'No hand gestures detected in the video'})
    
    # Apply smoothing to reduce noise - use a sliding window approach
    smoothed_predictions = smooth_predictions(predictions, confidence_scores, window_size=3)
    
    # Group consecutive identical predictions
    grouped_predictions = []
    if smoothed_predictions:
        current_gesture = smoothed_predictions[0]
        start_time = timestamps[0]
        end_time = timestamps[0]
        
        for i in range(1, len(smoothed_predictions)):
            if smoothed_predictions[i] == current_gesture:
                end_time = timestamps[i]
            else:
                grouped_predictions.append({
                    'gesture': current_gesture,
                    'start_time': round(start_time, 2),
                    'end_time': round(end_time, 2),
                    'duration': round(end_time - start_time, 2)
                })
                current_gesture = smoothed_predictions[i]
                start_time = timestamps[i]
                end_time = timestamps[i]
        
        # Add the last group
        grouped_predictions.append({
            'gesture': current_gesture,
            'start_time': round(start_time, 2),
            'end_time': round(end_time, 2),
            'duration': round(end_time - start_time, 2)
        })
    
    # Filter out very short duration predictions (likely noise)
    filtered_predictions = [p for p in grouped_predictions if p['duration'] >= 0.3]
    
    # Form words from the sequence of gestures
    formed_text = ""
    for pred in filtered_predictions:
        formed_text += pred['gesture'] + " "
    
    formed_text = formed_text.strip()
    
    # Find the most common gesture in the video
    if predictions:
        most_common = Counter(predictions).most_common(1)[0][0]
    else:
        most_common = None
    
    return jsonify({
        'video_duration': round(total_frames / fps, 2),
        'predictions': filtered_predictions,
        'formed_text': formed_text,
        'most_common_gesture': most_common,
        'raw_predictions': [{'gesture': p, 'time': round(t, 2), 'confidence': round(c, 2)} 
                           for p, t, c in zip(predictions, timestamps, confidence_scores)]
    })

def smooth_predictions(predictions, confidence_scores, window_size=3):
    """Apply smoothing to predictions using a sliding window approach"""
    if len(predictions) <= window_size:
        return predictions
    
    smoothed = []
    for i in range(len(predictions)):
        # Define window boundaries
        start = max(0, i - window_size // 2)
        end = min(len(predictions), i + window_size // 2 + 1)
        
        # Get predictions and confidences in the window
        window_preds = predictions[start:end]
        window_conf = confidence_scores[start:end]
        
        # Count occurrences weighted by confidence
        weighted_counts = {}
        for pred, conf in zip(window_preds, window_conf):
            if pred not in weighted_counts:
                weighted_counts[pred] = 0
            weighted_counts[pred] += conf
        
        # Select the prediction with highest weighted count
        smoothed.append(max(weighted_counts.items(), key=lambda x: x[1])[0])
    
    return smoothed

def analyze_hand_landmarks_for_video(results):
    """Similar to analyze_hand_landmarks but returns a dict instead of a JSON response"""
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

        # Get prediction and probability
        prediction = model.predict([np.asarray(data_aux)])
        try:
            # Try to get prediction probabilities if available
            proba = model.predict_proba([np.asarray(data_aux)])
            confidence = max(proba[0])
        except:
            confidence = 0.8  # Default confidence if predict_proba is not available
            
        predicted_character = labels_dict[int(prediction[0])]

        return {'predicted_gesture': predicted_character, 'confidence': confidence}
    
    return {'error': 'No hand detected'}

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
        predicted_character = labels_dict[int(prediction[0])]

        return jsonify({'predicted_gesture': predicted_character})
    
    return jsonify({'error': 'No hand detected'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Sign Language Recognition API',
        'endpoints': {
            '/predict': 'POST - Send an image or video file to get the predicted sign'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
