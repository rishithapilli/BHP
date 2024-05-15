from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load
    
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from the form
        CRIM = float(request.form['CRIM'])
        ZN = float(request.form['ZN'])
        INDUS = float(request.form['INDUS'])
        CHAS = float(request.form['CHAS'])
        NOX = float(request.form['NOX'])
        RM = float(request.form['RM'])
        Age = float(request.form['Age'])
        DIS = float(request.form['DIS'])
        RAD = float(request.form['RAD'])
        TAX = float(request.form['TAX'])
        PTRATIO = float(request.form['PTRATIO'])
        B = float(request.form['B'])
        LSTAT = float(request.form['LSTAT'])


        # Convert categorical variables using label encoders
        # Geography = label_encoders['Geography'].transform([Geography])[0]
        # Gender = label_encoders['Gender'].transform([Gender])[0]

        # Create DataFrame with user input
        data = {
            'CRIM': [CRIM],
            'ZN': [ZN],
            'INDUS': [INDUS],
            'CHAS': [CHAS],
            'NOX': [NOX],
            'RM': [RM],
            'Age': [Age],
            'DIS': [DIS],
            'RAD': [RAD],
            'TAX': [TAX],
            'PTRATIO':[PTRATIO],
            'B':[B],
            'LSTAT':[LSTAT]

        }

        df = pd.DataFrame(data)

        # Transform data using the scalar (assuming numerical features)
        # new_data = scalar.transform(np.array(list(df.values())).reshape(1, -1))
        new_data = scalar.transform(df.values.reshape(1, -1))


        # Make prediction
        prediction = regmodel.predict(new_data)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)




































































# import json
# import pickle

# from flask import Flask,request,app,jsonify,url_for,render_template
# import numpy as np
# import pandas as pd

# app=Flask(__name__)
# ## Load the model
# regmodel=pickle.load(open('regmodel.pkl','rb'))
# scalar=pickle.load(open('scaling.pkl','rb'))
# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

# @app.route('/predict',methods=['POST'])
# def predict():
#     data=[float(x) for x in request.form.values()]
#     final_input=scalar.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output=regmodel.predict(final_input)[0]
#     return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



# if __name__=="__main__":
#     app.run(debug=True)


##############################################################################################

# import pickle
# from flask import Flask,request,app,jsonify,url_for,render_template
# import numpy as np 
# import pandas as pd

# #flask app
# app = Flask(__name__ )

# #load the model
# regmodel = pickle.load(open('regmodel.pkl','rb'))

# scalar = pickle.load(open('scaling.pkl','rb'))

# #once the code hit the flask it redirect the html home
# #use to redirect the home 
# @app.route('/')
# def home():
#     return render_template('home.html')

# # predict API send request and get the output, give inputs 
# @app.route('/predict_api',methods=['POST'])
# #stand the normal units 
# def predict_api():
#     data = request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

# if __name__=="__main__":
#     app.run(debug=True)

