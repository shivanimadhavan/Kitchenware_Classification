1. Initial UI Window
● Initially, the interface displays the option for choosing an image using the "Choose an
Image" button. The picture below shows the interface for choosing the image where
no image is selected.
● By clicking on this "Choose an Image" button, the user can upload an image for
classification.
![image](https://github.com/user-attachments/assets/b333f6b2-dd96-4fb8-95ca-a3623eb7c999)
2. After Selecting an Image and Making a Prediction
● The system processes the image selected by the user.
● Then the category predicted by the model is displayed along with a bar chart below.
● The bar chart shows predicted probabilities for each kitchenware classes.
![image](https://github.com/user-attachments/assets/0f09b030-8693-425a-a3b4-55d2baa701c0)
3. Feedback Option:
● Feedback for dropdown and an "Update Model" checkbox are shown. To share the
feedback of prediction, you can use the "feedback" dropdown.
● The "Update Model" userbox is checked, meaning that the user wants to retrain the
model using this input.
● This step gives the system feedback so it can improve at predicting future outcomes.
![image](https://github.com/user-attachments/assets/ae10b408-c83b-4dae-8d40-9f957a3a8637)
4. Model Retraining in Progress
● When the user provides feedback, if you have checked on "Update Model" this
causes re-training of the model with the particular feedback.
● After that, the system shows a message "Training the model...” to indicate that the
model is being updated on the user side.
● The bar chart now displays that the model thinks category 3 is most likely (top row in
green) and it got the correct answer on this question (bottom row in blue).
![image](https://github.com/user-attachments/assets/f7644b12-aa61-4e32-80fa-86aaec5a09b8)






