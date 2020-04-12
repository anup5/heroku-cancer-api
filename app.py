import pickle
import numpy as np
from flask import Flask, request, render_template
import gunicorn

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final_features = [np.array(input_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    return render_template('index.html', prediction_text='Predicted class is : {} '.format(output))


if __name__ == "__main__":
    app.run(debug=True)
