import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Resizing, Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.callbacks import TensorBoard  # Import TensorBoard
import datetime  # To name logs uniquely
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from tensorflow.keras.optimizers import RMSprop
# Constants for image dimensions and training
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 26
EPOCHS = 25

# Paths for training and testing datasets
# Prompt for base directory
BASE_DIR = input("Enter the base directory (or press Enter to use the current directory): ").strip() or os.getcwd()

# Dynamically prompt the user to input train and test directories, resolving relative paths
TRAIN_DIR = input("Enter the path for the train directory: ").strip()
if not os.path.isabs(TRAIN_DIR):
    TRAIN_DIR = os.path.join(BASE_DIR, TRAIN_DIR)

TEST_DIR = input("Enter the path for the test directory: ").strip()
if not os.path.isabs(TEST_DIR):
    TEST_DIR = os.path.join(BASE_DIR, TEST_DIR)

# Ensure directories exist
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

# Set up log directory for TensorBoard
log_dir = os.path.join(BASE_DIR, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

def is_valid_image(file_path):
    """
    Checks if the file is a valid image format.

    Parameters:
        file_path (str): The path to the image file.

    Returns:
        bool: True if the file is a valid image, False otherwise.
    """
    try:
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img)
        return True
    except:
        return False

def clean_directory(directory):
    """
    Removes non-image files from the directory.

    Parameters:
        directory (str): The path to the directory to clean.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_valid_image(file_path):
                print(f"Removing invalid image file: {file_path}")
                os.remove(file_path)

# Clean directories
clean_directory(TRAIN_DIR)
clean_directory(TEST_DIR)

# Load the training, validation, and test datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    color_mode='rgb',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    batch_size=BATCH_SIZE,
    seed=42
)

# Get class names from the training dataset
class_labels = train_data.class_names

# Modify the function to build the CNN model
def build_cnn_model_with_augmentation(input_dim, num_classes):
    """
    Constructs and compiles a convolutional neural network model with data augmentation.
    
    Parameters:
        input_dim (tuple): Dimensions of the input images.
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): Compiled CNN model with data augmentation.
    """
    model = Sequential([
        # Data Augmentation Layer
        RandomFlip("horizontal_and_vertical", input_shape=input_dim),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2),
        RandomBrightness(0.2),  # New addition
        #RandomShear(0.2),  # New addition
        # Resizing and Rescaling
        Resizing(IMAGE_HEIGHT, IMAGE_WIDTH),
        Rescaling(1.0 / 255),

        # First Convolution Block
        Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # Second Convolution Block
        Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # Final Convolution Block
        Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),

        # Dense Layers
        Dense(1500, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    #model.compile(optimizer='adam',
    #loss='sparse_categorical_crossentropy',
    #metrics=['accuracy'])
    
    #return model
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize the enhanced model
input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
cnn_model_with_augmentation = build_cnn_model_with_augmentation(input_shape, len(class_labels))

# Train the model on the training dataset and validate on the validation dataset
def train_model(model, train_dataset, val_dataset, epochs, callbacks=[]):
    """
    Trains the model on the provided training and validation datasets.
    
    Parameters:
        model (tf.keras.Model): The CNN model to train.
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        epochs (int): Number of epochs for training.
        callbacks (list): List of callbacks (e.g., TensorBoard).
    
    Returns:
        history (tf.keras.callbacks.History): Training history object.
    """
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)
    return history

# Train the model with TensorBoard callback
training_history_with_aug = train_model(cnn_model_with_augmentation, train_data, validation_data, EPOCHS, [tensorboard_callback])

# Evaluate the model on the test dataset
def evaluate_model(model, test_dataset):
    """
    Evaluates the model on the test dataset.
    
    Parameters:
        model (tf.keras.Model): The CNN model to evaluate.
        test_dataset (tf.data.Dataset): The test dataset.
    
    Returns:
        test_accuracy (float): Accuracy of the model on the test dataset.
    """
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=2)
    print(f'\nTest accuracy: {test_accuracy:.4f}')
    return test_accuracy

test_accuracy_with_aug = evaluate_model(cnn_model_with_augmentation, test_data)

# Plot the training and validation accuracy and loss
def plot_training_results(history, epochs):
    """
    Plots the training and validation accuracy and loss over epochs.
    
    Parameters:
        history (tf.keras.callbacks.History): Training history object.
        epochs (int): Number of epochs for training.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(epochs)

    plt.figure(figsize=(12, 5))

    # Plotting Training Accuracy
    plt.subplot(1, 2, 1)
    plt.bar(epoch_range, acc, label='Training Accuracy', color='red', alpha=0.6, width=0.4, align='center')
    plt.bar(epoch_range, val_acc, label='Validation Accuracy', color='black', alpha=0.6, width=0.4, align='edge')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plotting Training and Validation Loss using bar charts
    plt.subplot(1, 2, 2)
    plt.bar(epoch_range, loss, label='Training Loss', color='orange', alpha=0.6, width=0.4, align='center')
    plt.bar(epoch_range, val_loss, label='Validation Loss', color='green', alpha=0.6, width=0.4, align='edge')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()

plot_training_results(training_history_with_aug, EPOCHS)


# Dynamically create a path for saving the model
save_path = os.path.join(BASE_DIR, 'TrainedModel', 'kitchenware_trainedmodel.keras')

# Save the trained model
def save_trained_model(model, save_path):
    """
    Saves the trained model to the specified path.
    
    Parameters:
        model (tf.keras.Model): The trained model to save.
        save_path (str): Path where the model will be saved.
    """
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the model in the specified path
    model.save(save_path)
    print(f'Model saved to {save_path}')

save_trained_model(cnn_model_with_augmentation, save_path)