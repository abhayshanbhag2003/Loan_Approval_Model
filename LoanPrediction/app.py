from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
from LoanPrediction import Encoder

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('Loan.html')


@app.route('/predict', methods=['POST'])
def predict():
    final_features = []

    # iterate over the form inputs
    for key in request.form:
        # append the input value to the list
        feature=[]
        feature.append(request.form[key])
        final_features.append(feature)

    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]


    for c in final_features:
        c = Encoder.fit_transform(c)
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)

    output =prediction[0]
    if(output>=0.5):
        return render_template('Loan.html',prediction_text='LOAN APPROVED ')
    else:
        return render_template('Loan.html',prediction_text='LOAN NOT APPROVED ')
if __name__ == '__main__':
    app.run(debug=True)
