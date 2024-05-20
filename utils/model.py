import pandas as pd
import pickle

# Load Model
def load(path: str):
    with open(path, "rb") as file:
        return pickle.load(file)

def predict(data, model):
    # Ubah data menjadi Pandas Dataframe
    df = pd.DataFrame(data)
    
    pred = model.predict(df)
    proba = model.predict_proba(df)[0]
    classes = model.classes_
    
    chart_data = pd.DataFrame(proba, classes)
    
    return pred, chart_data