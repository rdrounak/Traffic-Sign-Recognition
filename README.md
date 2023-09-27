# Traffic Sign Recognition System with CNN

## Introduction

This Traffic Sign Recognition System is a computer vision project that uses Convolutional Neural Networks (CNNs) to recognize and classify traffic signs in real-time. It includes both a Graphical User Interface (GUI) for image recognition and a live video capturing method. This README provides an overview of the project, its features, and instructions on how to set it up and use it.

## Features

- Real-time traffic sign recognition using a pre-trained CNN model.
- GUI for easy image recognition without any coding knowledge.
- Live video feed for continuous traffic sign recognition.
- High accuracy in recognizing various traffic signs.
- Interactive and user-friendly interface.

## Prerequisites

Before using this Traffic Sign Recognition System, ensure you have the following prerequisites installed:

- Python (3.7 or higher)
- OpenCV
- TensorFlow
- Numpy

You can install these dependencies using pip:

```bash
pip install opencv-python tensorflow  numpy
```

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/rdrounak/Traffic-Sign-Recognition.git
cd traffic-sign-recognition
```

### 2. Run the GUI

To use the GUI for traffic sign recognition:

```bash
python gui.py
```

The GUI will open, allowing you to either select an image or use your computer's webcam for real-time traffic sign recognition.

### 3. Live Video Capture

To use the live video capture for traffic sign recognition:

```bash
python TrafficSign_Test.py
```

This will open a window showing your webcam feed with real-time traffic sign recognition displayed on the video.

## Model

The CNN model used for traffic sign recognition is pre-trained and included in the repository. You can find it in the `models` directory.

## Training (Optional)

If you wish to train your own CNN model or fine-tune the existing model, you can use the dataset provided in the `dataset` directory. You can use popular deep learning frameworks like TensorFlow or PyTorch to train and save your model.

## Credits

This project is developed by Rounak Dwary. It is based on deep learning techniques and libraries provided by the open-source community.


## Acknowledgments

Special thanks to the open-source community for providing the tools and datasets necessary for this project.

Feel free to contribute, report issues, or suggest improvements to this project. Happy traffic sign recognition!
