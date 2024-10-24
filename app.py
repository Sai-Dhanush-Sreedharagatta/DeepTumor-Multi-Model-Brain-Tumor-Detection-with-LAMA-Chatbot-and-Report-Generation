# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for
import random
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Create the Flask app
app = Flask(__name__)
MODEL_PATH = r'D:\GMU\FALL2024\AIT 736\Final_project\brain_tumor_detection_model.keras'

# Load the trained model
model = load_model(MODEL_PATH)

# Define the classes based on your model training
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
else:
    print("Model loaded successfully!")

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file is in the request
    if 'file' not in request.files:
        return redirect(request.url)
    
    # Get the file from the request
    file = request.files['file']

    if file:
        # Save the uploaded image to a temporary location
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)  # Create folder if it doesn't exist
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Preprocess the image and make a prediction
        img = image.load_img(file_path, target_size=(224, 224))  # Ensure the target size matches your model's input size
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class, file_path='uploads/' + file.filename)

# Route for selecting random sample images
@app.route('/random', methods=['GET'])
def random_sample():
    # List of sample images stored in the static folder
    sample_images_folder = 'static/sample_images'
    sample_images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg']
    selected_image = random.choice(sample_images)
    file_path = os.path.join(sample_images_folder, selected_image)

    # Preprocess the image and make a prediction
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class, file_path='sample_images/' + selected_image)

if __name__ == '__main__':
    app.run(debug=True)
