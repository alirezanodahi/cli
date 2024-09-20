# Skin Cancer Detection using Deep Learning

## Overview
This project implements an IoT-based skin cancer detection system using deep learning techniques. The model classifies skin lesions as either benign (nevus or seborrheic keratosis) or malignant (melanoma) based on images. The system utilizes a pre-trained InceptionV3 model for feature extraction and binary classification.

## Dataset
The dataset is obtained from [Udacity's Dermatologist AI](https://github.com/udacity/dermatologist-ai) and consists of three parts:
- **Training set**: Contains images for training the model.
- **Validation set**: Used to validate the model's performance during training.
- **Testing set**: Used for final evaluation of the model's performance.

## Project Structure
- `data/`: Directory containing the downloaded dataset.
- `train.csv`: Metadata file containing paths and labels for training data.
- `valid.csv`: Metadata file for validation data.
- `test.csv`: Metadata file for testing data.
- Python scripts for data preprocessing, model training, evaluation, and visualization.

## Requirements
- TensorFlow
- TensorFlow Hub
- Matplotlib
- NumPy
- Pandas
- Seaborn
- Scikit-learn
- imbalanced-learn

## Setup Instructions

1. **Download and Extract Dataset**: 
   - The dataset can be downloaded from the links provided in the code. The script automatically extracts the dataset into the `data/` directory.

2. **Generate CSV Metadata**:
   - CSV files are generated for the training, validation, and testing datasets, which map image file paths to their respective labels.

3. **Load and Preprocess Data**:
   - The images are loaded and preprocessed, including resizing and normalization.

4. **Prepare Datasets**:
   - Datasets are prepared for training and validation with appropriate batching and shuffling.

5. **Build and Train Model**:
   - The model architecture is built using the InceptionV3 model as a feature extractor, followed by a dense layer for classification. The model is trained using the training dataset.

6. **Evaluate Model**:
   - The trained model is evaluated on the testing dataset, and metrics such as accuracy, sensitivity, and specificity are calculated.

7. **Visualize Results**:
   - Confusion matrices and ROC curves are plotted for performance evaluation, and a few test images are displayed with predicted labels.

8. **Predict Image Class**:
   - A function is provided to predict the class of a new image, displaying the prediction result alongside the image.

## Usage
- To predict the class of a specific image, use the `predict_image_class` function with the path to the image and the trained model.

## Future Enhancements
- Integration of additional image classification algorithms.
- Use of more advanced models for improved accuracy.
- Implementation of a web interface for user interaction.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or suggestions.
