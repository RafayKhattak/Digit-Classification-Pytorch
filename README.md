# MNIST Handwritten Digit Classification with PyTorch
This project demonstrates a simple implementation of a deep learning model for classifying handwritten digits from the MNIST dataset using the PyTorch library. The MNIST dataset is a widely-used benchmark dataset in the field of computer vision.
## Project Overview
The goal of this project is to train a convolutional neural network (CNN) model to accurately classify handwritten digits from the MNIST dataset. The model is built using PyTorch, a popular deep learning framework, and trained using the Adam optimizer.

The project involves the following steps:
- Loading and preprocessing the MNIST dataset
- Designing and building a CNN model architecture
- Training the model on the training data
- Evaluating the model's performance on the test data
- Saving and loading the trained model
- Performing inference on new images
## Requirements
- Python (3.x)
- PyTorch (1.x)
- torchvision
- PIL
## Installation
1. Clone the repository:
```
git clone https://github.com/your-username/mnist-classification-pytorch.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
## Usage
1. Prepare the dataset:
- The MNIST dataset will be automatically downloaded and preprocessed during the first run of the script. However, if you want to specify a different data directory or adjust any preprocessing parameters, you can modify the configuration in the script.
2. Train the model:
- Run the training script to train the model. You can adjust hyperparameters such as the number of epochs, learning rate, and batch size in the script.
3. Evaluate the model:
- After training, the model's performance on the test set will be evaluated automatically, and the accuracy score will be displayed.
4. Perform inference:
- You can use the trained model to make predictions on new images by running the inference script and providing the path to the image file.
