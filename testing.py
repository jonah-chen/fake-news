import pandas as pd
df = pd.read_csv("test_data/fake.csv")
df = df["title"]

from preprocessing import to_npy, to_npy_2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    X = np.array(list(executor.map(to_npy_2,df[:200])),dtype=np.float32)
from tensorflow.keras.models import load_model
model = load_model("model-title.h5")
Y = model(X, training=False)
print(Y)
wrong = 0
for [e1, e2] in Y:
    if (e1 > e2):
        wrong += 1
print(wrong)