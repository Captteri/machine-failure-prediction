from flask import Flask, render_template, request, jsonify
import joblib
app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load("machine_failure_lg.joblib")

@app.route('/')
def index():
    return render_template('index.html')

# Define a route for the home page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        
        # Assume data is a dictionary with feature values
        features = [
            float(data['air_temperature']),
            float(data['process_temperature']),
            float(data['rotational_speed']),
            float(data['torque'])
        ]

        # Convert features to numpy array and make a prediction
        prediction = loaded_model.predict([features])

       # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

    
    



