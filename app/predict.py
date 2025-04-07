import joblib
import numpy as np 

model = joblib.load ("app/model.joblib")

def predict_iris(features: list):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return int(prediction[0])
       