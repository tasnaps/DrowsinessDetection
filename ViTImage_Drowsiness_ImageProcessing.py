import argparse
import cv2
import torch
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification
from PIL import Image
from collections import deque

# Load the model and image processor
model_name = "chbh7051/vit-driver-drowsiness-detection"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)


# Define a function that processes either an image or a frame from video
def process_image(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence_score = probabilities[0][predicted_class].item()
    return predicted_class, confidence_score


# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Process an image or video for drowsiness detection.')
parser.add_argument('path', type=str, help='The path to the image or video file')
args = parser.parse_args()

# Use a deque to store the last N predicted classes and confidence scores, where N is the window size for moving average
window_size = 10
moving_avg_preds_and_scores = deque(maxlen=window_size)

# Check the file extension
if args.path.lower().endswith(('.png', '.jpg', '.jpeg')):
    # Process an image
    image = Image.open(args.path)
    predicted_class, confidence_score = process_image(image)
    moving_avg_preds_and_scores.append((predicted_class, confidence_score))
elif args.path.lower().endswith(('.mp4', '.avi', '.mov', '.flv', '.wmv')):
    # Process a video
    cap = cv2.VideoCapture(args.path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            predicted_class, confidence_score = process_image(image)
            moving_avg_preds_and_scores.append((predicted_class, confidence_score))
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Invalid file type. Please provide an image (png, jpg, jpeg) or video (mp4, avi, mov, flv, wmv) file.")

# Print the overall predicted class (0 for awake, 1 for drowsy) and its confidence score, based on moving average
avg_predicted_class = round(sum(x[0] for x in moving_avg_preds_and_scores) / len(moving_avg_preds_and_scores))
avg_confidence_score = sum(x[1] for x in moving_avg_preds_and_scores) / len(moving_avg_preds_and_scores)
print(f"Predicted class: {avg_predicted_class}, Confidence score: {avg_confidence_score}")
if avg_predicted_class == 0:
    print("Subject Awake")
else:
    print("Subject Drowsy")
