import json
import pickle
from flask import Flask, render_template,request, app,jsonify, url_for
import numpy as np
import pandas as pd

app=Flask(__name__)

# Load the model
bst_model=pickle.load(open("model.pkl",'rb'))

@app.route('/',methods=['GET'])

def home():
    return render_template('telecom-churn.html')

@app.route("/predict",methods=['POST'])

def predict():
    input_features=[float(x) for x in request.form.values()]
    data=[np.array(input_features).reshape(-1)]
    output=bst_model.predict(data)
    def zen(x):
        if x==1.0:
            s='Yes'
        else:
            s='No'
        return s

    Answer=zen(output[0])
    
    
    return render_template('telecom-churn.html',prediction_text="Will Customer Churn ?  {}".format(Answer))

if __name__=="__main__":
    app.run(debug=True)
