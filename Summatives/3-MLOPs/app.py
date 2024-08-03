#!/usr/bin/python3
"""
This app uses a neural network model to predict the risk of death 
from heart failure based on user input.

The app has two routes:
    - GET '/' - Displays the home page with a form for user input.
    - POST '/predict' - Makes a prediction based on the user input
      and displays the result.
    - POST '/retrain' - Retrains the model based on new data 
      uploaded by the user.
exepcted inputs: 
    - age
    - creatinine_phosphokinase
    - diabetes
    - ejection_fraction
    - high_blood_pressure
    - platelets
    - serum_creatinine
    - time
Return: Returns a prediction and risk level based on the user input.
"""

from flask import Flask, render_template, request, jsonify
import pickle as pkl
import os
import logging
from tensorflow import keras
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder="src/templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True

logging.basicConfig(level=logging.INFO)

# Load the preprocessor and model
try:
    preprocessor = joblib.load('pipeline/preprocessor.pkl')
    model = keras.models.load_model('models/model.h5')
except Exception as e:
    logging.error(f"Error loading model or preprocessor: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
        predict function:
        - predicts the risk of death from heart failure based on user input.
    """
    
    if request.method == 'POST':
        try:
            prediction = None
            risk_level = None
            features = {key: float(request.form[key]) for key in request.form.keys()}
            
            logging.info("Features collected: %s", features)
            
            input_data = pd.DataFrame([features])
            
            input_data_preprocessed = preprocessor.transform(input_data)
            logging.info("Preprocessed input data: %s", input_data_preprocessed)
            
            prediction = model.predict(input_data_preprocessed)[0][0]
            logging.info("Prediction: %s", prediction)
            
            risk_level = "Low" if prediction < 0.33 else "Moderate" if prediction < 0.67 else "High"
            percentage_prediction = f"{prediction * 100:.2f}"
            return render_template('result.html', prediction=percentage_prediction, risk_level=risk_level)
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            return render_template('index.html', error="Invalid input. Please enter valid numbers.")
        except Exception as e:
            logging.error(f"Exception: {e}")
            return render_template('index.html', error="An error occurred. Please try again.")
    
    return render_template('index.html')

@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    """ 
    retrain function:
    - retrain the model based on new data uploaded by the user.
    """
    if request.method == 'POST':
        try:
            file = request.files['datafile']
            if not file:
                return "No file uploaded", 400
            
            new_data = pd.read_csv(file)
            logging.info("New data loaded for retraining: %s", new_data.head())
            
            X = new_data.drop('DEATH_EVENT', axis=1)
            y = new_data['DEATH_EVENT']
            
            X = X.drop(['sex', 'serum_sodium', 'smoking', 'anaemia'], axis=1)

            X_processed = preprocessor.transform(X)

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_processed, y)

            def build_model(input_shape):
                model = keras.Sequential([
                    keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model

            model = build_model(X_resampled.shape[1])
            model.fit(X_resampled, y_resampled, epochs=100, batch_size=32)

            model.save('models/model.h5')
            joblib.dump(preprocessor, 'pipeline/preprocessor.pkl')

            return render_template('index.html'), 200
        except Exception as e:
            logging.error(f"Error during retraining: {e}")
            return "An error occurred during retraining.", 500

    return render_template('retrain.html')

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if request.method == 'POST':
        try:
            file = request.files['datafile']
            if not file:
                return "No file uploaded", 400
            
            new_data = pd.read_csv(file)
            logging.info("New data loaded for evaluation: %s", new_data.head())
            
            required_columns = ['age', 'creatinine_phosphokinase', 'diabetes', 
                                'ejection_fraction', 'high_blood_pressure', 
                                'platelets', 'serum_creatinine', 'time', 'DEATH_EVENT']
            
            for col in required_columns:
                if col not in new_data.columns:
                    return f"Missing required column: {col}", 400
            
            X = new_data[required_columns[:-1]]
            y = new_data['DEATH_EVENT']
            
            X_processed = preprocessor.transform(X)

            predictions = model.predict(X_processed)
            predictions = (predictions > 0.5).astype(int)

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions)
            recall = recall_score(y, predictions)
            f1 = f1_score(y, predictions)

            return render_template('evaluate.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            return f"An error occurred during evaluation: {str(e)}", 500
    
    return render_template('evaluate.html')





if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
