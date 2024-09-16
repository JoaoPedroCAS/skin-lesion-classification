import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.applications import vgg19, inception_v3, xception
from pathlib import Path
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import load_img, img_to_array
import cv2

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paths
DRIVE_PATH = Path('INSERTYOURPATH') ### ---> INSERT YOUR PATH HERE
DATA_FILE_PATH = DRIVE_PATH / 'data.txt' ### ---> DATA.TXT CONTAIS THE NAME OF EACH IMAGE TO BE EVALUATED AND THE CLASS IT BELONGS
DATA_DIR = DRIVE_PATH / 'data' ### ---> INSIDE THE DATA FOLDER YOU WILL FIND ALL THE IMAGES FROM THE HAM10000 DATASET  
RESULTS_PATH = DRIVE_PATH / 'results.txt' ### ---> WILL SAVE THE RESULTS HERE

# Models
AVAILABLE_MODELS = {
    "VGG19": vgg19.VGG19,
    "Inception": inception_v3.InceptionV3,
    "Xception": xception.Xception
}

def build_model(base_model, num_classes):
    """
    Constructs a model by adding a custom classification head to a given base model.

    Args:
        base_model (tf.keras.Model): The base model to use as a feature extractor. 
            The base model's layers will be frozen during training.
        num_classes (int): The number of output classes for classification. 

    Returns:
        tf.keras.Model: A Keras Model instance that combines the base model with a custom classification head.

    The custom classification head consists of:
        - GlobalAveragePooling2D layer to reduce spatial dimensions.
        - Dense layer with 512 units and ReLU activation function.
        - BatchNormalization layer to improve training speed and stability.
        - Dropout layer with a rate of 0.5 to prevent overfitting.
        - Dense layer with 256 units and ReLU activation function.
        - BatchNormalization layer to improve training speed and stability.
        - Dropout layer with a rate of 0.5 to prevent overfitting.
        - Dense output layer with 'num_classes' units and softmax activation function for classification.

    The final model takes the input from the base model and outputs the predictions from the classification head.
    """
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)          # Pooling layer for reducing spatial dimensions
    x = Dense(512, activation='relu')(x)     # Smaller Dense layer
    x = BatchNormalization()(x)              # Batch normalization for faster convergence
    x = Dropout(0.5)(x)                      # Dropout to prevent overfitting
    x = Dense(256, activation='relu')(x)     # Another Dense layer
    x = BatchNormalization()(x)              # Batch normalization after Dense layer
    x = Dropout(0.5)(x)                      # Dropout to prevent overfitting
    predictions = Dense(num_classes, activation='softmax')(x)  # Output layer with softmax
    
    return Model(inputs=base_model.input, outputs=predictions)

def train_model(model, X_train, y_train, X_val, y_val, patience=10):
    """
    Trains a Keras model on the provided training data with early stopping based on validation accuracy.

    Args:
        model (tf.keras.Model): The Keras model to be trained.
        train (tf.data.Dataset or tf.keras.utils.Sequence): The training dataset. Should be a `tf.data.Dataset` or 
            `tf.keras.utils.Sequence` instance providing the input data and labels.
        val (tf.data.Dataset or tf.keras.utils.Sequence): The validation dataset. Should be a `tf.data.Dataset` or 
            `tf.keras.utils.Sequence` instance providing the input data and labels.
        patience (int, optional): Number of epochs to wait for improvement in validation accuracy before stopping. Default is 10.

    Returns:
        tf.keras.callbacks.History: A History object containing details of the training process, including loss and 
            accuracy metrics for both training and validation data.

    The model is compiled with the following configuration:
        - Optimizer: Adam
        - Loss function: Categorical Crossentropy
        - Metrics: Accuracy

    The training process uses early stopping to halt training when validation accuracy stops improving for a specified number of epochs (patience).
    The training process is verbose with a progress bar shown during each epoch.
    """
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

    return model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a Keras model on the test dataset.

    Args:
        model (tf.keras.Model): The trained Keras model to be evaluated.
        X_test (np.ndarray): The input features of the test dataset.
        y_test (np.ndarray): The true labels of the test dataset, one-hot encoded.

    Returns:
        dict: A dictionary containing performance metrics:
            - "Accuracy": Accuracy score of the model.
            - "Precision": Weighted precision score of the model.
            - "Recall": Weighted recall score of the model.
            - "F1-Score": Weighted F1 score of the model.
            - "ROC AUC": ROC AUC score for multi-class classification.

    The function performs the following steps:
        - Predicts the probabilities for the test dataset.
        - Converts the predicted probabilities to class predictions.
        - Converts the one-hot encoded true labels to class labels.
        - Computes accuracy, precision, recall, and F1-score using weighted averages.
        - Computes the ROC AUC score for multi-class classification.
        - Prints the classification report and confusion matrix.

    Note:
        - The precision, recall, and F1-score calculations use `average='weighted'` and handle division by zero with `zero_division=1`.
        - The ROC AUC score is calculated using `average='macro'` and `multi_class='ovr'`.
    """

    # Get predicted probabilities and predicted classes
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=1)
    
    # Calculate ROC AUC score for multi-class classification
    roc_auc = roc_auc_score(y_test, y_pred, average='macro', multi_class='ovr')

    # Print the classification report and confusion matrix
    print(classification_report(y_true, y_pred_classes))
    print(confusion_matrix(y_true, y_pred_classes))

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc,
    }

def load_and_preprocess_images(image_paths, target_size):
    """
    Loads and preprocesses images from file paths.

    Args:
        image_paths (list of str): List of file paths to images.
        target_size (tuple of int): Target size to resize images to (height, width).

    Returns:
        np.ndarray: Array of preprocessed images.
    """
    images = []
    for path in image_paths:
        img = load_img(path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = tf.keras.applications.vgg19.preprocess_input(img_array)  # Change preprocessing if using a different model
        images.append(img_array)
    return np.array(images)


# Initialize lists for storing data
X = []
y = []

# Metrics storage
accuracies = []
precisions = []
f1_scores = []
roc_aucs = []

# Read data from file and process each line
with open(DATA_FILE_PATH, 'r') as file:
    for line in file:
        # Split line into name and class_label
        name, class_label = line.split()
        
        # Construct image path and read image
        img_path = DATA_DIR / name
        img = cv2.imread(str(img_path))
        
        # Resize image and append to list
        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(class_label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Convert labels to categorical
num_classes = len(le.classes_)
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

selected_model_name = input("Enter model name (VGG19, Inception ou Xception): ")
selected_model = AVAILABLE_MODELS[selected_model_name](weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Split the data into training, testing and validation sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Build the model
model = build_model(selected_model, num_classes)
# Train the model
history = train_model(model, X_train, y_train, X_val, y_val)
# Evaluate the model
evaluation = evaluate_model(model, X_test, y_test)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Accuracy - {selected_model_name}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

# Save results
with open(RESULTS_PATH, 'w') as f:
    f.write(f"Accuracy: {evaluation['Accuracy']:.4f}\n")
    f.write(f"Precision: {evaluation['Precision']:.4f}\n")
    f.write(f"Recall: {evaluation['Recall']:.4f}\n")
    f.write(f"F1-Score: {evaluation['F1-Score']:.4f}\n")
    f.write(f"ROC AUC: {evaluation['ROC AUC']:.4f}\n")