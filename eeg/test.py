import numpy as np
import pandas as pd
from joblib import load

df = pd.read_csv("data.csv", index_col=0)
labels = df.pop("y")
labels[labels != 1] = 0

pcas = [df @ np.load(f".\\pca\\pca{i}.npy") for i in range(2, 6)]


models = [load(f".\\models\\rf\\1v{i}.joblib") for i in range(2, 6)]

predictions = [model.predict(pca) for model, pca in zip(models, pcas)]
# y_pred = sum(predictions) <= len(predictions) // 2
# diagnosis = "Seizure" if y_pred else "No seizure"
# diagnosis = "-".join(predictions)
print(predictions)
