from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='C:/Users/mehra/OneDrive/Documents/GitHub/boston-housing-project')
import numpy as np
from tensorflow import keras

model = keras.models.load_model("boston_housing.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    float_features = [float(x) for x in request.form.values()]
    final_features = np.array(float_features).reshape(1,-1)
    prediction = model.predict(final_features).tolist()
    print(prediction)
    return render_template('index.html', prediction_text='The House Price Should Be {} USD'.format(round(prediction[0][0] * 1000),2))


if __name__ == "__main__":
    app.run(debug=True)