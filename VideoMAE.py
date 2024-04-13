"""
VideoMAE.py
Author: tasnaps
Date: 13 April 2024
"""


import os
import argparse
import time
import cv2
import torch
import numpy as np
from collections import deque
from torch import softmax
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# Load the model and image processor
model_name = "kyle0518/videomae-base-fatigue-detection-full"
model = VideoMAEForVideoClassification.from_pretrained(model_name)
processor = VideoMAEImageProcessor.from_pretrained(model_name)
os.environ['TORCH_WARN_ASSUME_LIST_NOT_LARGE'] = '1'
# The expected number of frames.
num_frames = 16


def process_images(images):
    # Convert list of images to numpy array
    images = np.array(images)

    # Convert colors from BGR to RGB for all images
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

    # Resize images to 224x224 for the model
    images_resized = [cv2.resize(img_rgb, (224, 224)) for img_rgb in images_rgb]

    # Divide by 255.0 to normalize pixel values to range [0, 1]
    images_normalized = [img_resized / 255.0 for img_resized in images_resized]

    # Combine all frames together
    images_ready = np.stack(images_normalized)

    # Change dimension from (num_frames, height, width, channels) to (num_frames, channels, height, width)
    images_ready = np.transpose(images_ready, (0, 3, 1, 2))

    # Add batch dimension and convert to float
    images_ready = np.expand_dims(images_ready, axis=0).astype(np.float32)

    # Create inputs for the model
    inputs = {"pixel_values": torch.tensor(images_ready)}

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


parser = argparse.ArgumentParser(description='Process an image or video for detection.')
parser.add_argument('path', type=str, help='The path to the image or video file')
args = parser.parse_args()

last_frames = deque(maxlen=num_frames)

if args.path.lower().endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv')):
    cap = cv2.VideoCapture(args.path)

    if cap.isOpened():
        print("Video file opened successfully.")
    else:
        print("Failed to open video file.")
        exit(1)

    frameCount = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            last_frames.append(frame)
            frameCount += 1
            if len(last_frames) == num_frames:
                print(f"Processing frame set starting at frame #{frameCount - num_frames + 1}")
                predicted_class, confidence_score = process_images(np.array(list(last_frames)))

                if predicted_class == 0:
                    print(f"At current set of frames, subject is Awake with confidence score of {confidence_score}")
                else:
                    print(f"At current set of frames, subject is Drowsy with confidence score of {confidence_score}")

                last_frames.clear()

            time.sleep(0.1)

        else:
            print(f"Video ended after {frameCount} frames.")
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Invalid file type. Please provide a video file.")
