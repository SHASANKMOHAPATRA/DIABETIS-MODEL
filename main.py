

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('diabetes_model.sav','rb'))

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    features = data['features']  # Assuming features are passed as a list
    
    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)
    
    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
