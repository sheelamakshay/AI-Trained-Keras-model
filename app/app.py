from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import traceback

app = Flask(__name__)

# Absolute path to your model
MODEL_PATH = r"C:\Users\rlais\Downloads\rice_leaf_disease_model.keras"
tf.keras.models.load_model(MODEL_PATH)

# Define the labels
labels_dict = {0: 'Bacterial leaf blight', 1: 'Brown spot', 2: 'Leaf smut'}

# Define the image size your model expects
IMG_SIZE = (180, 180)

def prepare_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize as done in training
        return img
    except Exception as e:
        print(f"Error preparing image: {e}")
        traceback.print_exc()
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    try:
        if request.method == 'POST':
            # Check if a file is uploaded
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file:
                # Save the file to a specified absolute location
                file_path = os.path.join("static/uploads", file.filename)
                file.save(file_path)

                # Prepare the image and predict
                img = prepare_image(file_path)
                if img is None:
                    return "Error processing the image.", 500
                
                prediction = model.predict(img)
                predicted_class = np.argmax(prediction)

                # Get the label for the predicted class
                result = labels_dict[predicted_class]

                return render_template("result.html", result=result, image=file.filename)

        return render_template("index.html")
    
    except Exception as e:
        print(f"Error in upload_image function: {e}")
        traceback.print_exc()
        return "An error occurred.", 500

if __name__ == '__main__':
    app.run(debug=True)
