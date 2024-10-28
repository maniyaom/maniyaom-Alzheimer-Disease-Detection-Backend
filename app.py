from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

with open('model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

app = Flask(__name__)

# Enable CORS for all routes and allow all origins
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_percentage():
    try:
        data = request.json

        required_features = [
            "age", "gender", "ethnicity", "educationalLevel", "bmi", 
            "smoking", "alcoholConsumption", "physicalActivity", "dietQuality", 
            "sleepQuality", "familyHistoryAlzheimers", "cardiovascularDisease", 
            "diabetes", "depression", "headInjury", "hypertension", 
            "systolicBP", "diastolicBP", "cholesterolTotal", "cholesterolLDL", 
            "cholesterolHDL", "cholesterolTriglycerides", "mmse", 
            "functionalAssessment", "memoryComplaints", "behavioralProblems", 
            "adl", "confusion", "personalityChanges", 
            "difficultyCompletingTasks", "forgetfulness"
        ]

        if len(required_features) != len(data):
            return jsonify({'error': 'Data must contain all required features.'}), 400

        features = pd.DataFrame([data])
        features_array = features.values
        percentage_prediction = rf_model.predict_proba(features_array)[0][1] * 100
        
        return jsonify({'prediction': round(percentage_prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
if __name__ == '__main__':
    app.run(debug=True)
