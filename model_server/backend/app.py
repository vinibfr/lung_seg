from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
from model import predict as model  # Import your actual Python model code
from flask_cors import CORS  # Import the CORS module
import base64
import numpy as np
app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    file.save(file.filename)
    return 'File uploaded successfully', 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        image = request.files['image']

        # Load the image and process it with your model
        npy_array = np.load(image)
        
        return_image = model(npy_array)  # Adjust based on your model interface

        return send_file(return_image, mimetype='image/png', as_attachment=True)

    else:
        return jsonify({'error': 'No image provided'})

if __name__ == '__main__':
    app.run(debug=True)
