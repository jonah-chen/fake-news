import pandas as pd
import seaborn as scn
import numpy as np

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake.drop(columns=["date", "subject"], inplace=True)
real.drop(columns=["date", "subject"], inplace=True)

df.sample(n = 3) 
