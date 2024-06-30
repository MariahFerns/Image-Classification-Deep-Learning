# 🚤Image Classification using Deep Learning

## Introduction
This project aims to build an automatic system to classify different types of boats using deep learning techniques. The goal is to reduce human error in misclassifying boat types by developing a bias-free and corruption-free system. The project involves creating a Convolutional Neural Network (CNN) and using transfer learning to build a lightweight model for deployment on mobile devices.

## Objective
- Develop a CNN model to classify boat images into 9 different categories.
- Build a lightweight model using transfer learning for mobile deployment.
- Compare the performance of both models.

## Dataset Description
The dataset contains images of 9 types of boats:
- Buoy
- Cruise_ship
- Ferry_boat
- Freight_boat
- Gondola
- Inflatable_boat
- Kayak
- Paper_boat
- Sailboat

The dataset consists of a total of 1162 images, organized into directories corresponding to each class.

## Steps and Tasks Performed
### 1. Data Preparation
- Load and preprocess the dataset.
- Split the data into training and testing sets.

### 2. Build and Train a CNN Model
- Construct a CNN architecture for boat classification.
- Compile the model using Adam optimizer, categorical_crossentropy loss, and metrics such as accuracy, precision, and recall.
- Train the model on the training data.
- Evaluate the model on the test data.
- Plot the confusion matrix and generate a classification report.

### 3. Transfer Learning for Lightweight Model
- Use a pre-trained MobileNetV2 model as the base.
- Add custom layers on top of MobileNetV2 for classification.
- Compile the model with similar metrics as the CNN model.
- Implement early stopping during training.
- Convert the trained model to TensorFlow Lite for mobile deployment.

### 4. Comparison and Evaluation
- Compare the performance of the CNN model and the MobileNetV2-based model.
- Analyze metrics such as accuracy, precision, recall, and F1-score.

## Model Architectures
### CNN Model from scratch
- Data augmentation by flipping, rotation and zoom and data normalization.
- Convolutional layers with ReLU activation.
- MaxPooling layers to reduce dimensionality.
- Dropout layers for regularization.
- Dense layers with softmax activation for classification.

### MobileNetV2-based Model using transfer learning
- Pre-trained MobileNetV2 as the base model.
- Custom dense layers added on top for classification.
- Early stopping during training to prevent overfitting.

### Reduce number of layers in above transfer learning model

### Fine tuning the pre-trained model
- Unfreezing the MobileNetv2 model and retraininng it.

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MariahFerns/Image-Classification-Deep-Learning.git
   cd Image-Classification-Deep-Learning

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt

3. Run Jupyter Notebooks:
   Navigate to the notebooks/ folder and open the notebooks to explore data analysis and model development.

## Results and Findings
- Using a pre-trained model like MobileNetV2 gives much better accuracy than training a model from scratch.
- Adding fewer layers on top of a pre-trained model gives better performance as it reduces complexity and prevents overfitting.
- Fine-tuning the pre-trained model may not always lead to better performance; it should be tried as per the use case at hand.

## Conclusion
This project successfully developed a bias-free automatic boat classification system. The CNN model and the MobileNetV2-based model both demonstrated good performance, with the latter being more accurate and suitable for mobile deployment due to its lightweight architecture. This system can help reduce human error in classifying boat types and improve operational efficiency in port regions.
