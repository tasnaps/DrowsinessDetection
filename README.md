# Video and Image Analysis using ViT / VideoMAE

This project was made for University of Eastern Finland course: Advanced Human Computer Interaction course.

We followed these steps:
Record videos for a period of days, at the same time do a subjective KSS (Karolinska Sleepiness Scale) analysis.
Analyze all the videos with a script: VideoMAE.py
Later using the MainAnalyzer we analyzed the results - what we were looking for was correlation between scores and model analysis.

The given  model gave awake biased classifications, and correlation between KSS and Model analysis remained weak.




MainAnalyzer.py script uses a pretrained deep learning model to analyze a series of video files, detecting and classifying whether the individual in each video appears to be awake or drowsy. This analysis is performed on a frame-by-frame basis, prioritizing computational efficiency by processing batches of frames concurrently. The results of these classifications, including counts and confidence-weighted percentages for the awake and drowsy states, are logged and saved to a CSV file for easy review and further analysis.

## Dependencies

- Python 3
- OpenCV
- PyTorch
- NumPy
- Pandas
- Transformers (Hugging Face)

## Core Functions

### `process_images(images)`

The `process_images` function takes an array of images (or a batch of video frames) and processes them for analysis. Image processing steps include:

1. Converting the color scheme from BGR to RGB.
2. Resizing the images to the dimensions required by the pretrained model.
3. Normalizing pixel values to a [0, 1] range.
4. Rearranging dimensions to accommodate the model's input expectations.
5. Adding a batch dimension and converting data types for compatibility with PyTorch tensors.

Once the images have been appropriately processed, they're fed into the pretrained model for classification. The model's output logits are then converted into probability distributions via the softmax function. The function subsequently returns the most likely classification ('awake' or 'drowsy') and its associated confidence level.

### `analyze_video(video_path: str)`

The `analyze_video` function takes a path to a video file as input and processes the video file frame by frame. Frames are kept in a queue (the `last_frames` deque), with a set length equal to the number of frames processed concurrently by the model (`num_frames`). Once this length is reached (i.e., there are enough frames to create a batch), these frames are passed together into the `process_images` function, and their associated classification and confidence score is tallied. This process continues until all frames in the video have been analyzed.

The function also calculates two types of percentages for each state (awake and drowsy): one based on the count of frames classified in that state and one based on the total confidence scores tabulated for that state across all batches.

Additionally, it calculates the average and standard deviation of all confidence scores obtained during the analysis of the video.

### `analyze_videos()`

This function iterates through all the video files in a specified directory, sends each video file to the `analyze_video` function for processing, and stores the results in a dictionary. The dictionaries are then collected together and converted into a pandas DataFrame. 

Before saving, the data is adjusted. Any columns specified in the `cols_to_format` list are rounded to a specific number of decimal places (8, in this case). This prepped DataFrame is then written to a CSV file. The column headers in the CSV file include, among others, the count and confidence-weighted percentages for each state (awake and drowsy), as well as the mean and standard deviation of all confidence scores. 

The `analyze_videos` function also reformats the date and time embedded in each video's filename into a more readable format to provide an easier understanding of when the videos were taken.



## Installation

Clone this repository to your local machine and install the necessary python dependencies by running:
bash git clone https://github.com/tasnaps/DrowsinessDetection 
cd DrowsinessDetection 
pip install -r requirements.txt

## Files

The project contains two main python files:

- `VideoMAE.py`: This is a script that processes video data using the VideoMAE model. It specifically handles video data streams and can be used for tasks that span several frames such as fatigue detection.

- `ViTImage_Drowsiness_ImageProcessing.py` : This script processes static image data using ViT model. It receives an image file path as an argument and outputs the results of the analysis.

## Usage

For video analysis, run the following command: bash python VideoMAE.py <video-file-path>

For image analysis, run: bash python ViTImage_Drowsiness_ImageProcessing.py <image-file-path>

Replace `<video-file-path>` and `<image-file-path>` with the path to your input video or image file.

## Dependencies

Please refer to the `requirements.txt` file for the required dependencies.

## Contributing

If you want to contribute to this project, please fork the repository and propose your changes by opening a pull request.

## License

This project is licensed under the terms of the MIT license.
