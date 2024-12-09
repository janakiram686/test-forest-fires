from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
 It creates an instance of the flask class,
 which will be your WSGI application

'''
### WSGI application
application=Flask(__name__)
app=application

# import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scalar=pickle.load(open('models/scalar.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temperature = float(request.form.get('temperature'))
        rh = float(request.form.get('rh'))
        ws = float(request.form.get('ws'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc =  float(request.form.get('dmc'))
        
        isi =  float(request.form.get('isi'))
       
        
        classes=  float(request.form.get('classes'))
        region =   float(request.form.get('region'))

        newdata_scaled=standard_scalar.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result=ridge_model.predict(newdata_scaled)

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)