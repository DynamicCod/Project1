from flask import Flask, request, jsonify
import joblib

# Load the trained model (make sure 'fake_news_model.pkl' is in the same directory)
model = joblib.load('fake_news_model.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data sent from the request
    data = request.get_json()

    # Extract text content from the data
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Make prediction using the loaded model
    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0][prediction]

    # Return the prediction and confidence
    return jsonify({
        'prediction': 'real' if prediction == 1 else 'fake',
        'confidence': float(probability)
    })

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)