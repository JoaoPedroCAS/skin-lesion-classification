# CNN CLASSIFIER

This repository contains code to classify skin lesions using images from the HAM10000 dataset. The code uses pre-trained models (VGG19, InceptionV3, Xception) as feature extractors, adds custom classification layers, and evaluates the model's performance.

---

## DEPENDENCIES 

Make sure to install the following dependencies before running the code: 

`pip install tensorflow keras numpy sklearn matplotlib opencv-python`

---

## DATASETS

This project uses the HAM10000 dataset, which is a collection of skin lesion images used for training models in dermatology. You need to download the dataset and place the images in a `data/` folder.

The `data.txt` file should contain the names of images and their corresponding class labels. Each line of `data.txt` is structured as follows:

`image_name.jpg class_label`

- `image_name.jpg` - The filename of the image in the `data/` folder.
- `class_label` - The integer representing the class label.

---

## PRETRAINED MODELS

The project supports the following pretrained models from Keras Applications:

- VGG19
- INCEPTION_V3
- XCEPTION

You can select one of these models during runtime. The chosen model will be used as a feature extractor, and the classification head is customized based on the number of classes in the dataset

---

## MODEL ARCHITECTURE

### BASE MODEL

The base models (VGG19, InceptionV3, or Xception) are initialized with ImageNet weights and their layers are frozen, meaning they are not updated during training.

### CUSTOM CLASSIFICATION HEAD

The custom classification head consists of:

- `GlobalAveragePooling2D`: Reduces the spatial dimensions of the base model's output.
- Dense Layer (512 units, ReLU activation)
- `BatchNormalization`: Normalizes activations to speed up convergence.
- `Dropout` (rate=0.5): Reduces overfitting by randomly dropping connections.
- Dense Layer (256 units, ReLU activation)
- `BatchNormalization`
- `Dropout` (rate=0.5)
- Output Layer: Number of units equals the number of classes, with softmax activation for classification.

---

## TRAINING

### Early Stopping

Training uses early stopping to avoid overfitting. If validation accuracy does not improve for a specified number of epochs (`patience=10`), training stops and the model reverts to the best weights.

### Train/Validation/Test Split

The dataset is split as follows:

- **80%** of data is split into training and validation sets (80% training, 20% validation).
- **20%** is reserved for testing.

### Model Compilation

The model is compiled with:

- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Metrics: `accuracy`

---

## MODEL TRAINING

The model is trained for a maximum of 100 epochs, with early stopping enabled.

---

## EVALUATION

Once the model is trained, it is evaluated on the test set using the following metrics:

- **Accuracy**: Measures the proportion of correctly classified instances.
- **Precision** (Weighted): Measures how many of the predicted positives are actual positives.
- **Recall** (Weighted): Measures how many of the actual positives were predicted correctly.
- **F1-Score** (Weighted): Harmonic mean of precision and recall.
- **ROC AUC**: Measures the area under the receiver operating characteristic curve for multi-class classification.

The code outputs the following:

- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **Confusion Matrix**: Shows the true vs. predicted class distributions.

Evaluation results are saved in `results.txt`.

---

## RESULTS

### RESULTS FOR VGG 19

| **Metric**    | **Mean** | **Standard Deviation (SD)** |
| ------------- | -------- | --------------------------- |
| **Accuracy**  | 0.7541   | 0.0042                      |
| **Precision** | 0.7552   | 0.0021                      |
| **F1-Score**  | 0.7449   | 0.0039                      |
| **ROC AUC**   | 0.9148   | 0.0012                      |

![VGG19](https://github.com/user-attachments/assets/d4e6adac-b59a-4c23-a680-829a06fe3263)


