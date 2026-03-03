import os
import base64
import json
from flask import Flask, render_template, request, jsonify
from utils.classifier import classify_image, get_model_metrics
from utils.gradcam import generate_gradcam_overlay
from utils.demo_images import get_demo_image

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify():
    model_name = request.form.get('model', 'resnet50')
    demo_type  = request.form.get('demo', None)

    # Demo image path (no real upload needed)
    if demo_type:
        result = classify_image(None, model_name, demo_type=demo_type)
        gradcam_b64 = generate_gradcam_overlay(demo_type, result['predicted_class'])
        result['gradcam'] = gradcam_b64
        return jsonify(result)

    # Real upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if f.filename == '' or not allowed_file(f.filename):
        return jsonify({'error': 'Invalid file. Use PNG or JPG.'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, 'upload.jpg')
    f.save(filepath)
    result = classify_image(filepath, model_name)
    gradcam_b64 = generate_gradcam_overlay('upload', result['predicted_class'])
    result['gradcam'] = gradcam_b64
    return jsonify(result)

@app.route('/api/metrics')
def metrics():
    return jsonify(get_model_metrics())

@app.route('/api/demo_image/<demo_type>')
def demo_image(demo_type):
    img_b64 = get_demo_image(demo_type)
    return jsonify({'image': img_b64})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
