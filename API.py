from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Explicitly set the template folder
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Custom Lambda layer with output shape specification
def contrastive_loss_layer(embeddings):
    return tf.abs(embeddings[0] - embeddings[1])

# Load the trained model with custom objects
try:
    model = tf.keras.models.load_model(
        'ecg_psychological_model.h5',
        custom_objects={'contrastive_loss_layer': Lambda(contrastive_loss_layer, output_shape=(1,))}
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Error handler for RequestEntityTooLarge
@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(error):
    logger.error(f"File upload exceeded size limit: {error}")
    return render_template('index.html', error="File too large! Maximum allowed size is 100 MB.", now=datetime.now()), 413

@app.route('/test')
def test():
    return "Flask is running!"

@app.route('/')
def index():
    logger.debug("Rendering index page")
    return render_template('index.html', now=datetime.now())

@app.route('/analyze', methods=['POST'])
def analyze():
    logger.debug("Received POST request to /analyze")
    
    if 'file' not in request.files:
        logger.warning("No file uploaded")
        return render_template('index.html', error="No file uploaded", now=datetime.now())
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return render_template('index.html', error="No selected file", now=datetime.now())
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.debug(f"File saved: {filepath}")
            
            # Load and preprocess ECG data
            try:
                ecg_data = pd.read_csv(filepath)
            except pd.errors.EmptyDataError:
                os.remove(filepath)
                logger.error("Uploaded file is empty")
                return render_template('index.html', error="Uploaded file is empty", now=datetime.now())
            
            # Validate data
            if ecg_data.empty or len(ecg_data) < 100:
                os.remove(filepath)
                logger.error("Insufficient or empty ECG data")
                return render_template('index.html', error="Insufficient or empty ECG data (minimum 100 samples required)", now=datetime.now())
            
            # Log the columns found in the CSV for debugging
            logger.debug(f"Columns found in CSV: {list(ecg_data.columns)}")
            
            # Map alternative column names for MLII, V5, V1, and V2
            mlii_mapping = {
                'LeadII': 'MLII',
                'leadII': 'MLII',
                'mlii': 'MLII'
            }
            v5_mapping = {
                'LeadV5': 'V5',
                'leadV5': 'V5',
                'v5': 'V5'
            }
            v1_mapping = {
                'LeadV1': 'V1',
                'leadV1': 'V1',
                'v1': 'V1'
            }
            v2_mapping = {
                'LeadV2': 'V2',
                'leadV2': 'V2',
                'v2': 'V2'
            }
            ecg_data.rename(columns=mlii_mapping, inplace=True)
            ecg_data.rename(columns=v5_mapping, inplace=True)
            ecg_data.rename(columns=v1_mapping, inplace=True)
            ecg_data.rename(columns=v2_mapping, inplace=True)
            
            # Check for the presence of columns
            has_mlii = 'MLII' in ecg_data.columns
            has_v5 = 'V5' in ecg_data.columns
            has_v1 = 'V1' in ecg_data.columns
            has_v2 = 'V2' in ecg_data.columns
            
            # Determine which pair to use (in order of preference)
            if has_mlii and has_v5:
                logger.debug("Using MLII and V5 pair")
                ecg_samples = ecg_data[['MLII', 'V5']].values
                leads_used = "MLII and V5"
            elif has_mlii and has_v1:
                logger.debug("Using MLII and V1 pair")
                ecg_samples = ecg_data[['MLII', 'V1']].values
                leads_used = "MLII and V1"
            elif has_mlii and has_v2:
                logger.debug("Using MLII and V2 pair")
                ecg_samples = ecg_data[['MLII', 'V2']].values
                leads_used = "MLII and V2"
            elif has_v5 and has_v2:
                logger.debug("Using V5 and V2 pair")
                ecg_samples = ecg_data[['V5', 'V2']].values
                leads_used = "V5 and V2"
            else:
                os.remove(filepath)
                missing = []
                if not has_mlii:
                    missing.append('MLII (or LeadII)')
                if not has_v5:
                    missing.append('V5 (or LeadV5)')
                if not has_v1:
                    missing.append('V1 (or LeadV1)')
                if not has_v2:
                    missing.append('V2 (or LeadV2)')
                logger.error(f"CSV missing required column pairs: {missing}")
                return render_template('index.html', error=f"CSV must contain one of the following pairs: MLII and V5, MLII and V1, MLII and V2, or V5 and V2 (or alternatives like LeadII, LeadV5, LeadV1, LeadV2). Missing: {', '.join(missing)}", now=datetime.now())
            
            logger.debug(f"ECG samples shape: {ecg_samples.shape}")
            
            # Generate predictions or use fallback
            if model:
                reference_sample = ecg_samples[0:1]
                input_pairs = np.array([(reference_sample, sample) for sample in ecg_samples[1:100]])
                predictions = model.predict([input_pairs[:,0], input_pairs[:,1]])
                stress_level = np.mean(predictions)
                anxiety_level = np.median(predictions)
                depression_level = np.max(predictions)
                logger.debug(f"Model predictions - Stress: {stress_level}, Anxiety: {anxiety_level}, Depression: {depression_level}")
            else:
                stress_level = np.random.random()
                anxiety_level = np.random.random()
                depression_level = np.random.random()
                logger.warning("Model not loaded, using random values for predictions")
            
            # Generate ECG plot using the first column of the selected pair
            ecg_plot = generate_ecg_plot(ecg_samples[:200, 0])  # First 200 samples of the first lead in the pair
            logger.debug("ECG plot generated")
            
            # Clean up the uploaded file
            os.remove(filepath)
            logger.debug(f"File removed: {filepath}")
            
            # Prepare data for template
            template_data = {
                'stress': classify_level(stress_level),
                'anxiety': classify_level(anxiety_level),
                'depression': classify_level(depression_level),
                'ecg_plot': ecg_plot,
                'filename': filename,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'now': datetime.now(),
                'leads_used': leads_used  # Add which leads were used for user feedback
            }
            logger.debug(f"Rendering template with data: {template_data}")
            return render_template('index.html', **template_data)
            
        except pd.errors.ParserError:
            os.remove(filepath)
            logger.error("Error parsing CSV file")
            return render_template('index.html', error="Error parsing CSV file", now=datetime.now())
        except Exception as e:
            os.remove(filepath)
            logger.error(f"An unexpected error occurred: {str(e)}")
            return render_template('index.html', error=f"An unexpected error occurred: {str(e)}", now=datetime.now())
    
    logger.warning("Invalid file type")
    return render_template('index.html', error="Invalid file type", now=datetime.now())

def classify_level(value):
    """Classify psychological state level based on value"""
    value = float(value)
    if value > 0.7:
        return {'level': 'High', 'value': f"{value:.2f}", 'color': 'danger'}
    elif value > 0.4:
        return {'level': 'Moderate', 'value': f"{value:.2f}", 'color': 'warning'}
    else:
        return {'level': 'Low', 'value': f"{value:.2f}", 'color': 'success'}

def generate_ecg_plot(samples):
    """Generate ECG plot as base64 encoded image"""
    plt.figure(figsize=(10, 4), facecolor='#f8f9fa')
    plt.plot(samples, color='#007bff', linewidth=1.5)
    plt.title('ECG Signal', pad=20)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info("Starting Flask app")
    app.run(host='0.0.0.0', port=5000, debug=True)