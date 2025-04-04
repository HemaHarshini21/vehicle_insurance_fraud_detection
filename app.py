from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # To handle CORS issues
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__, template_folder='../frontend')  # Specify the path to the frontend folder
CORS(app)  # This allows cross-origin requests from any domain

# Define the paths for the model and encoders
model_path = r"C:\Users\91855\Documents\New folder\backend"
encoders_path = r"C:\Users\91855\Documents\New folder\backend"

# Load the model and encoders
try:
    print("Loading model and encoders...")
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    print("Model and encoders loaded successfully.")
except Exception as e:
    print(f"Error loading model and encoders: {e}")
    model = None
    encoders = None

@app.route("/")
def home():
    return render_template('index.html')  # This will now correctly find index.html in the frontend folder

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model or not encoders:
            raise Exception("Model or encoders not loaded.")

        # Get the request data (JSON)
        data = request.get_json()

        # Convert the incoming data into a pandas DataFrame
        input_data = pd.DataFrame([data])

        # Apply the label encoders to the categorical columns
        for col, le in encoders.items():
            if col in input_data:
                input_data[col] = le.transform(input_data[col])

        # Make a prediction
        prediction = model.predict(input_data)
        prediction_result = "Fraudulent" if prediction[0] == 1 else "Valid"

        # Return the prediction result as JSON
        return jsonify({"message": f"The claim is {prediction_result}.", "alert_class": "alert-success" if prediction_result == "Valid" else "alert-danger"})

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}", "alert_class": "alert-danger"}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)