from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash
import os
import logging
from gtts import gTTS
import null_values
import binary
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import re
import pyttsx3



# Models
binary_model = load_model('./binary.h5')
application_model = load_model('./Full_data_LSTM_app.h5')
attack_model = load_model('./final_attack_multiclass.h5')



app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.DEBUG)

uploaded_file_path = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_file_path
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = file.filename
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_file_path)
        return jsonify({'result': 'File successfully uploaded.'})

@app.route('/classify', methods=['POST'])
def classify_file():
    global uploaded_file_path
    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        return jsonify({'error': 'Please upload a file first.'}), 400

    # Perform initial classification
    classification_result, data = initial_classification(uploaded_file_path)
    logging.debug(f"Initial classification result: {classification_result}")

    binary_graph_path = generate_binary_graph(classification_result)

    if classification_result == 'normal':
        application_indices = classify_application(data)
        result_text = f"Normal traffic detected. Application types: {format_result(application_indices, 'normal')}."
        multi_type_graph_path = generate_graph(application_indices, 'normal')
    else:
        attack_indices = classify_attack(data)
        result_text = f"Abnormal traffic detected. Attack types: {format_result(attack_indices, 'abnormal')}."
        multi_type_graph_path = generate_graph(attack_indices, 'abnormal')

    logging.debug(f"Final result: {result_text}")
    return jsonify({'result': result_text, 'binaryGraphPath': url_for('graph', filename=os.path.basename(binary_graph_path)), 'multiTypeGraphPath': url_for('graph', filename=os.path.basename(multi_type_graph_path))})

def format_result(indices, result_type):
    if result_type == 'normal':
        types_list = ['Google', 'HTTP', 'HTTP-Proxy', 'SSL', 'HTTP-Connect', 'YouTube',
                      'Amazon', 'Microsoft', 'Apple-ICloud', 'Apple-Itunes', 'Spotify',
                      'IP-ICMP', 'Apple', 'DropBox', 'MS-One-Drive', 'Instagram',
                      "What's Up", 'Google-Maps', 'CloudFlare', 'Gmail', "MSN",
                      'Office-365', 'X (Twitter)', 'Windows-Update', 'Facebook', 'DNS',
                      'Wikipedia', 'Yahoo', 'Netflix', 'Ebay', 'Skype', 'Content-Flash']
    else:
        types_list = ['DoS Hulk', 'PortScan', 'DDoS', 'Dos GoldenEye',
                      'Heartbleed', 'FTP-Patator', 'SSH-Patator', 'Infiltration', 'Web XSS',
                      'DoS Slowloris', 'Bot', 'Web Brute Force', 'Web SQL Injection', 'DoS Slowhttptest']
    
    type_counts = np.bincount(indices, minlength=len(types_list))
    result_summary = ", ".join([f"{types_list[i]}: {count}" for i, count in enumerate(type_counts) if count > 0])
    
    return result_summary

@app.route('/process_voice_command', methods=['POST'])
def process_voice_command():
    data = request.json
    command = data['command']
    logging.debug(f"Received voice command: {command}")

    # Check if the file has been uploaded
    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        return jsonify({'voiceResponseUrl': generate_voice_response("Please upload a file first.")})

    # Process the command
    if re.search(r'classify', command, re.IGNORECASE):
        result_dict = classify_data(uploaded_file_path)
        result = result_dict['result']
        binary_graph_path = result_dict['binaryGraphPath']
        multi_type_graph_path = result_dict['multiTypeGraphPath']
        response_text = f"The data has been classified. The result is: {result}."
    else:
        response_text = "Command not recognized. Please say 'classify the data' to start classification."
        binary_graph_path = ""
        multi_type_graph_path = ""

    voice_response_url = generate_voice_response(response_text)
    return jsonify({'voiceResponseUrl': voice_response_url, 'result': response_text, 'binaryGraphPath': binary_graph_path, 'multiTypeGraphPath': multi_type_graph_path})

def initial_classification(file_path):
    dataset = pd.read_csv(file_path)
    data = preprocess_data(dataset)
    
    if 'L7Protocol' in data.columns:
        prediction = binary_model.predict(data.drop('L7Protocol', axis=1))
    else:
        prediction = binary_model.predict(data)

    prediction = pred(prediction)
    predic_list = [item for sublist in prediction for item in sublist]
    # result = all(item == 0 for item in predic_list)
    zero_one_count = np.bincount(predic_list)
    if zero_one_count[0] > zero_one_count[1]:
        return 'normal', data
    else:
        return 'abnormal', data

def pred(y_test):
    y_test = np.array(y_test)
    y_pred_binary = np.where(y_test > 0.5, 1, 0)
    return y_pred_binary

def classify_application(data):
    # check for miss feature if applicable
    
    if 'L7Protocol' in data.columns:
        data['L7Protocol_present'] = 1
    else:
        data['L7Protocol'] = 0
        data['L7Protocol_present'] = 0
        
    prediction = application_model.predict(data)
    predicted_indices = np.argmax(prediction, axis=1)
    return predicted_indices

def classify_attack(data):
    prediction = attack_model.predict(data)
    predicted_indices = np.argmax(prediction, axis=1)
    return predicted_indices

def preprocess_data(file_path):
        
    if file_path.columns[-1] in ['Label', ' Label', 'label', 'ProtocolName']:
        file_path.drop(file_path.columns[-1], axis=1, inplace=True)
    data = null_values.NullFeatures(file_path).null_val()
    return binary.Preprocess(data).scale_feature()

def classify_data(file_path):
    classification_result, data = initial_classification(file_path)
    logging.debug(f"Initial classification result: {classification_result}")

    binary_graph_path = generate_binary_graph(classification_result)

    if classification_result == 'normal':
        application_indices = classify_application(data)
        result_text = f"Normal traffic detected. Application types: {format_result(application_indices, 'normal')}."
        multi_type_graph_path = generate_graph(application_indices, 'normal')
    else:
        attack_indices = classify_attack(data)
        result_text = f"Abnormal traffic detected. Attack types: {format_result(attack_indices, 'abnormal')}."
        multi_type_graph_path = generate_graph(attack_indices, 'abnormal')

    logging.debug(f"Classification result: {result_text}")
    return {'result': result_text, 'binaryGraphPath': binary_graph_path, 'multiTypeGraphPath': multi_type_graph_path}

# Initialize pyttsx3 TTS engine
tts_engine = pyttsx3.init()

# Adjust TTS engine properties for a more natural voice
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

def generate_voice_response(text):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'response.mp3')
    tts_engine.save_to_file(text, file_path)
    tts_engine.runAndWait()
    return url_for('uploaded_file', filename='response.mp3')

@app.route('/graph/<filename>')
def graph(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def generate_graph(predictions, result_type):
    plt.figure(figsize=(6, 4))  # Increase the figure size to ensure all labels are visible
    
    if result_type == 'normal':
        application_types = ['Google', 'HTTP', 'HTTP-Proxy', 'SSL', 'HTTP-Connect', 'YouTube',
                             'Amazon', 'Microsoft', 'Apple-ICloud', 'Apple-Itunes', 'Spotify',
                             'IP-ICMP', 'Apple', 'DropBox', 'MS-One-Drive', 'Instagram',
                             "What's Up", 'Google-Maps', 'CloudFlare', 'Gmail', "MSN",
                             'Office-365', 'X (Twitter)', 'Windows-Update', 'Facebook', 'DNS',
                             'Wikipedia', 'Yahoo', 'Netflix', 'Ebay', 'Skype', 'Content-Flash']
        sns.countplot(x=[application_types[i] for i in predictions], palette='viridis')
        plt.title('Application Type Distribution')
        plt.xlabel('Application Type')
    else:
        attack_types = ['DoS Hulk', 'PortScan', 'DDoS', 'Dos GoldenEye',
                        'Heartbleed', 'FTP-Patator', 'SSH-Patator', 'Infiltration', 'Web XSS',
                        'DoS Slowloris', 'Bot', 'Web Brute Force', 'Web SQL Injection', 'DoS Slowhttptest']
        sns.countplot(x=[attack_types[i] for i in predictions], palette='viridis')
        plt.title('Attack Type Distribution')
        plt.xlabel('Attack Type')

    plt.xticks(rotation=90)  # Rotate labels for better visibility
    plt.ylabel('Count')
    plt.tight_layout()  # Adjust layout to ensure labels are not cut off

    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'multi_type_classification.png')
    plt.savefig(graph_path)
    plt.close()
    
    return graph_path

def generate_binary_graph(result):
    plt.figure(figsize=(6, 4))  # Increase the figure size to ensure all labels are visible
    
    if result == 'normal':    
        normal = 1
        abnormal = 0
    else:
        abnormal = 1
        normal = 0
    gen_data = pd.DataFrame({
        'Category': ['Normal', 'Ab-normal'],
        'Count': [normal, abnormal]        
    })
    sns.barplot(data = gen_data, x = 'Category', y = 'Count', hue = 'Category', palette='viridis', dodge=False, legend=False)
    plt.title('Binary Classification Result')
    plt.xlabel('Classification')
    plt.ylabel('Count')
    plt.tight_layout()  # Adjust layout to ensure labels are not cut off

    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'binary_classification.png')
    plt.savefig(graph_path)
    plt.close()
    
    return graph_path

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)