from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('artifacts/models/model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get data from form
        sepal_length = float(request.form['SepalLengthCm'])
        sepal_width = float(request.form['SepalWidthCm'])
        petal_length = float(request.form['PetalLengthCm'])
        petal_width = float(request.form['PetalWidthCm'])
        
        # Prepare the input for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction using the model
        prediction = model.predict(input_data)[0]
        
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)