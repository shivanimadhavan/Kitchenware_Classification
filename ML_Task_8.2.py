from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import sys
import tensorflow as tf
import numpy as np
import os
import shutil
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Global variables for configuration
image_height, image_width = 256, 256
batch_size = 26
epochs = 5
BASE_DIR = input("Enter the base directory (or press Enter to use the current directory): ").strip() or os.getcwd()

# Dynamically set paths for training and test datasets
train_data_path = input("Enter the path for the train directory: ").strip()
if not os.path.isabs(train_data_path):
    train_data_path = os.path.join(BASE_DIR, train_data_path)

test_data_path = input("Enter the path for the test directory: ").strip()
if not os.path.isabs(test_data_path):
    test_data_path = os.path.join(BASE_DIR, test_data_path)

# Ensure directories exist
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Train directory not found: {train_data_path}")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test directory not found: {test_data_path}")

# Main Window class for the PyQt5 application
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Kitchenware Classifier")
        self.setGeometry(200, 200, 800, 600)

        self.image_path = None
        self.preds = None  # Initialize an instance variable for predictions

        self.initUI()

    def initUI(self):
        # Create a button to open file dialog
        self.btn_open = QtWidgets.QPushButton(self)
        self.btn_open.setText("Choose an Image")
        self.btn_open.clicked.connect(self.open_image)
        self.btn_open.setGeometry(50, 50, 200, 50)

        # Label to display prediction result
        self.result_label = QtWidgets.QLabel(self)
        self.result_label.setText("")
        self.result_label.setGeometry(50, 120, 700, 30)

        # Feedback dropdown and update model checkbox
        self.feedback_combo = QtWidgets.QComboBox(self)
        self.feedback_combo.addItems(['No feedback', 'Correct', 'Should be "cups"', 'Should be "knife"', 'Should be "scissor"', 'Should be "largespoon"'])
        self.feedback_combo.setGeometry(50, 160, 200, 30)
        self.feedback_combo.hide()

        self.update_checkbox = QtWidgets.QCheckBox("Update Model", self)
        self.update_checkbox.setGeometry(300, 160, 200, 30)
        self.update_checkbox.hide()

        self.btn_submit = QtWidgets.QPushButton(self)
        self.btn_submit.setText("Submit Feedback")
        self.btn_submit.clicked.connect(self.submit_feedback)
        self.btn_submit.setGeometry(50, 200, 200, 50)
        self.btn_submit.hide()

        # Matplotlib canvas
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setGeometry(50, 300, 700, 300)
        self.canvas.setParent(self)

    def open_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if self.image_path:
            self.predict_image()

    def predict_image(self):
        # Preprocess the image
        img = cv2.imread(self.image_path)
        img_resized = cv2.resize(img, (image_width, image_height))
        img_expanded = np.expand_dims(img_resized, axis=0)

        # Prediction
        self.preds = model.predict(img_expanded)  # Store the predictions in an instance variable
        predicted_class = classes[np.argmax(self.preds)]
        result_text = f'Prediction: The image is of category "{predicted_class}"'
        
        self.result_label.setText(result_text)
        self.feedback_combo.show()
        self.update_checkbox.show()
        self.btn_submit.show()

        self.preds = np.squeeze(self.preds)
        self.plot_predictions(self.preds)

    def plot_predictions(self, probabilities, actual_class=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        bars = ax.bar(range(len(classes)), probabilities, color='skyblue', edgecolor='black', linewidth=1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(range(len(classes)))
        
        # Rotate the x-axis labels to prevent them from getting cut off
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
        
        ax.set_ylim([0, 1])
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_xlabel('Kitchenware Class', fontsize=12)
        ax.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')

        predicted_class = np.argmax(probabilities)
        bars[predicted_class].set_color('#FF9999')  # Light red for the predicted class

        if actual_class is not None:
            bars[actual_class].set_color('#66CC66')  # Light green for the actual class

        for bar, prob in zip(bars, probabilities):
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + 0.02, 
                f'{prob:.2f}', 
                ha='center', 
                fontsize=10, 
                color='black'
            )

        # Ensure everything fits into the figure without cutting off
        plt.tight_layout()
        self.canvas.draw()

    def submit_feedback(self):
        feedback = self.feedback_combo.currentText()
        model_update = self.update_checkbox.isChecked()
        expected_class_index = None

        if feedback == 'Correct':
            expected_class_index = np.argmax(self.preds)
        elif feedback == 'Should be "cups"':
            expected_class_index = 0
        elif feedback == 'Should be "knife"':
            expected_class_index = 1
        elif feedback == 'Should be "scissor"':
            expected_class_index = 2
        elif feedback == 'Should be "largespoon"':
            expected_class_index = 3

        if expected_class_index is not None:
            self.plot_predictions(self.preds, expected_class_index)
            self.feedback_combo.hide()
            self.update_checkbox.hide()
            self.btn_submit.hide()

            if model_update:
                user_image_path = os.path.join(BASE_DIR, 'userInput')
                
                if os.path.exists(user_image_path):
                    shutil.rmtree(user_image_path)
                os.makedirs(user_image_path, exist_ok=True)

                for cls_name in classes:
                    os.makedirs(os.path.join(user_image_path, cls_name), exist_ok=True)

                shutil.copy(self.image_path, os.path.join(user_image_path, classes[expected_class_index]))

                user_data = tf.keras.preprocessing.image_dataset_from_directory(
                    user_image_path,
                    color_mode='rgb',
                    image_size=(image_height, image_width),
                    batch_size=1,
                    seed=100
                )

                self.result_label.setText('Training the model...')
                QApplication.processEvents()
                model.fit(user_data, validation_data=validation_dataset, epochs=epochs)

        self.result_label.setText("Upload an image for prediction!")
        self.feedback_combo.hide()
        self.update_checkbox.hide()
        self.btn_submit.hide()

# Load model and datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_path,
    color_mode='rgb',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=100
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_path,
    color_mode='rgb',
    shuffle=True,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=100
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_path,
    color_mode='rgb',
    shuffle=True,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    seed=100
)

classes = test_dataset.class_names
num_classes = len(classes)

model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'TrainedModel', 'kitchenware_trainedmodel.keras'))
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=2)
print('\nTest Accuracy:', test_accuracy)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# Save the updated model
model.save(os.path.join(BASE_DIR, 'TrainedModel', 'kitchenware_trainedmodel.keras'))