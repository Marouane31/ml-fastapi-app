from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    iris = load_iris()
    X, y = iris.data, iris.target

    model = RandomForestClassifier()
    model.fit (X,y)

    joblib.dump (model, 'app/model.joblib')