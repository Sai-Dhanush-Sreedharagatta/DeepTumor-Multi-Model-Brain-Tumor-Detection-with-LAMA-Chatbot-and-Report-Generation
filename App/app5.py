from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import random

# Initialize the Flask app
app = Flask(__name__)

# Define the model path and load the model
MODEL_PATH = 'D:/AIT736/Project/AIT736_FinalProject-main/brain_tumor_detection_model.h5'
model = load_model(MODEL_PATH)

# Define the classes based on your model training
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Route to handle prediction from uploaded files
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file:
        # Save the uploaded file to a temporary location
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Preprocess the image and make a prediction
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class, file_path='uploads/' + file.filename)

# Route to handle prediction from sample images
@app.route('/sample', methods=['GET'])
def sample_predict():
    # Path to the sample images folder
    sample_images_folder = 'static/sample'
    sample_images = os.listdir(sample_images_folder)
    selected_image = random.choice(sample_images)
    file_path = os.path.join(sample_images_folder, selected_image)

    # Preprocess the selected sample image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class, file_path='sample/' + selected_image)

# Run the app
if __name__ == '__main__':
    app.run(port=3000, debug=True)
