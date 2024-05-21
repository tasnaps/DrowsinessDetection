"""
VideoMAE.py
Author: tasnaps
Date: 21 May 2024
"""
import os
import cv2
import torch
import numpy as np
import pandas as pd
from collections import deque
from torch import softmax
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from datetime import datetime

# If cuda:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the model and image processor
model_name = "kyle0518/videomae-base-fatigue-detection-full"
model = VideoMAEForVideoClassification.from_pretrained(model_name).to(device)
processor = VideoMAEImageProcessor.from_pretrained(model_name)
os.environ['TORCH_WARN_ASSUME_LIST_NOT_LARGE'] = '1'
num_frames = 16

def process_images(images):
    # Convert list of images to numpy array
    images = np.array(images)
    # Convert colors from BGR to RGB for all images
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    images_resized = [cv2.resize(img_rgb, (224, 224)) for img_rgb in images_rgb]
    images_normalized = [img_resized / 255.0 for img_resized in images_resized]
    # Combine all frames together
    images_ready = np.stack(images_normalized)
    # Change dimension from (num_frames, height, width, channels) to (num_frames, channels, height, width)
    images_ready = np.transpose(images_ready, (0, 3, 1, 2))
    # Add batch dimension and convert to float
    images_ready = np.expand_dims(images_ready, axis=0).astype(np.float32)
    # Create inputs for the model
    inputs = {"pixel_values": torch.tensor(images_ready).to(device)}
    # Get outputs from the model
    logits = model(**inputs).logits
    # Apply softmax function to logits to make it into probabilities
    probabilities = softmax(logits, dim=-1)
    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    # Fetch confidence_score for predicted_class
    confidence_score = probabilities[0, predicted_class].item()
    print(f'Predicted class: {predicted_class}, confidence score: {confidence_score}')
    return predicted_class, confidence_score

def analyze_video(video_path: str):
    drowsiness_count = 0
    awakeness_count = 0
    drowsiness_confidence = 0.0
    awakeness_confidence = 0.0
    last_frames = deque(maxlen=num_frames)
    confidence_scores = []
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            last_frames.append(frame)
            frame_count += 1
            if len(last_frames) == num_frames:
                predicted_class, confidence_score = process_images(np.array(list(last_frames)))
                confidence_scores.append(confidence_score)
                if predicted_class == 1:
                    awakeness_count += 1
                    awakeness_confidence += confidence_score
                elif predicted_class == 2:
                    drowsiness_count += 1
                    drowsiness_confidence += confidence_score
                last_frames.clear()
        else:
            break
    cap.release()

    total_predictions = awakeness_count + drowsiness_count
    total_confidence = awakeness_confidence + drowsiness_confidence

    if total_predictions > 0:
        awake_percent = (awakeness_count / total_predictions) * 100
        drowsy_percent = (drowsiness_count / total_predictions) * 100
    else:
        awake_percent = drowsy_percent = 0

    if total_confidence > 0:
        awake_conf_percent = (awakeness_confidence / total_confidence) * 100
        drowsy_conf_percent = (drowsiness_confidence / total_confidence) * 100
    else:
        awake_conf_percent = drowsy_conf_percent = 0

    mean_confidence = np.mean(confidence_scores) if confidence_scores else 0
    std_dev_confidence = np.std(confidence_scores) if confidence_scores else 0

    return (awakeness_count, drowsiness_count, awake_percent, drowsy_percent,
            awake_conf_percent, drowsy_conf_percent, mean_confidence, std_dev_confidence)

# Get all the video files in the specified directory and process them
video_dir = "C:\\Users\\tapio\\Desktop\\Videos"
def analyze_videos():
    video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))
                   and f.lower().endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv'))]

    data = []
    for filename in video_files:
        video_path = os.path.join(video_dir, filename)
        date, time, kss = filename.split('_')[1:4]
        kss = kss.replace('KSS', '').replace('.mp4', '')

        # Convert date and time to more readable format
        date_time = datetime.strptime(date + time, '%Y%m%d%H%M%S')
        formatted_date = date_time.strftime('%Y-%m-%d')
        formatted_time = date_time.strftime('%H:%M:%S')

        (awakeness_count, drowsiness_count, awake_percent, drowsy_percent,
         awake_conf_percent, drowsy_conf_percent, mean_confidence, std_dev_confidence) = analyze_video(video_path)

        data.append({
            'Date': formatted_date,
            'Time': formatted_time,
            'KSS': kss,
            'User': 'Stephan',
            'Awakeness Count': awakeness_count,
            'Drowsiness Count': drowsiness_count,
            'Awakeness %': awake_percent,
            'Drowsiness %': drowsy_percent,
            'Awakeness Confidence %': awake_conf_percent,
            'Drowsiness Confidence %': drowsy_conf_percent,
            'Mean Confidence': mean_confidence,
            'Standard Deviation Confidence': std_dev_confidence
        })

    df = pd.DataFrame(data)
    cols_to_format = ['Awakeness %', 'Drowsiness %', 'Awakeness Confidence %',
                      'Drowsiness Confidence %', 'Mean Confidence', 'Standard Deviation Confidence']

    # Apply rounding to four decimal points
    df[cols_to_format] = df[cols_to_format].round(8)
    df.to_csv("Stephan_results.csv", index=False)

analyze_videos()
