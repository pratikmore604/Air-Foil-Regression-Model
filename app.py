import pandas as pd
import numpy as np
import  pickle
from flask import Flask,request,app,jsonify,url,render_template

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/predict_api",methods=["POST"])
def pedict_api():

    data = request.json["data"]
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]

    return jsonify(output)

@app.route("/predict",methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_feature = [np.array(data)]
    print(data)

    output = model.predict(final_feature)[0]

    return render_template("home.html",prediction_text="Airfoil pressure is {}".format(output))

if __name__ == "__main__":
    app.run(debug = True)