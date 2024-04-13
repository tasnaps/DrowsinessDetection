# Video and Image Analysis using ViT / VideoMAE

This project implements an analysis of video and image data using the Vision Transformer (ViT) model and the VideoMAE model. It allows processing video streams and static images for various types of detection tasks.

## Installation

Clone this repository to your local machine and install the necessary python dependencies by running:
bash git clone https://github.com/tasnaps/DrowsinessDetection cd DrowsinessDetection pip install -r requirements.txt

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
