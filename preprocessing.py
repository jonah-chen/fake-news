import pandas as pd
import matplotlib.pyplot as plt
import seaborn as scn
import numpy as np

import spacy
nlp = spacy.load("en_core_web_lg")

from time import perf_counter

from concurrent.futures import ProcessPoolExecutor

MAX_LEN = 1000 # longest title is 54, longest text 9469
MAX_LEN_TITLE = 54
PADDING_CONST = -1e8

def max_len(sentence):
    return len(nlp(sentence))

def to_npy(sentence):
    """Converts spacy array to numpy arrays

    Args:
        A (spacy.tokens.doc.doc): Array of word vector


    Returns:
        np.array(len(A), 300): Word vector as np array
    """
    A = nlp(sentence)

    arr = np.empty((MAX_LEN,300,), dtype=np.float32)

    if len(A) < MAX_LEN:
        for i in range(len(A)):
            arr[i] = A[i].vector
        for i in range(len(A), MAX_LEN):
            arr[i] = MAX_LEN*np.ones((300))
    
    else:
        for i in range(MAX_LEN):
            arr[i] = A[i].vector

    return arr

def to_npy_2(sentence):
    """Converts spacy array to numpy arrays (for the titles)

    Args:
        A (spacy.tokens.doc.doc): Array of word vector


    Returns:
        np.array(len(A), 300): Word vector as np array
    """
    A = nlp(sentence)

    arr = np.empty((MAX_LEN_TITLE,300,), dtype=np.float32)

    if len(A) < MAX_LEN_TITLE:
        for i in range(len(A)):
            arr[i] = A[i].vector
        for i in range(len(A), MAX_LEN_TITLE):
            arr[i] = MAX_LEN_TITLE*np.ones((300))
    
    else:
        for i in range(MAX_LEN_TITLE):
            arr[i] = A[i].vector

    return arr

# with ProcessPoolExecutor(max_workers=24) as executor:
#     x_train = list(executor.map(max_len, fake["text"]))

# # Don't forget about the fake as well
# print(max(x_train))
# print(np.std(np.array(x_train)))

def generate_body(end=None):
    """Generate the word vectors for the body of real and fake news articles

    Args:
        end (int, optional): The number of elements of real and fake news. Defaults to None.

    Returns:
        (np.array, np.array): real, fake
    """
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    fake.drop(columns=["date", "subject"], inplace=True)
    real.drop(columns=["date", "subject"], inplace=True)
    with ProcessPoolExecutor(max_workers=24) as executor:
        x_real = np.array(list(executor.map(to_npy, real["text"][:end])), dtype=np.float32)
        x_fake = np.array(list(executor.map(to_npy, fake["text"][:end])), dtype=np.float32)
    return x_real, x_fake

def generate_title(end=None):
    """Generate the word vectors for the titles of real and fake news articles

    Args:
        end (int, optional): The number of elements of real and fake news. Defaults to None.

    Returns:
        (np.array, np.array): real, fake
    """
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    fake.drop(columns=["date", "subject"], inplace=True)
    real.drop(columns=["date", "subject"], inplace=True)
    with ProcessPoolExecutor(max_workers=24) as executor:
        x_real = np.array(list(executor.map(to_npy_2, real["title"][:end])), dtype=np.float32)
        x_fake = np.array(list(executor.map(to_npy_2, fake["title"][:end])), dtype=np.float32)
    return x_real, x_fake

def get_dataset_2():
    """Generate the data for the second dataset

    Returns:
        np.array,np.array: X and Y
    """
    df = pd.read_csv("data/news.csv")
    df = df.dropna()
    X = df["text"]
    Y = df["label"]
    with ProcessPoolExecutor() as executor:
        X_t2 = np.array(list(executor.map(to_npy,X)),dtype=np.float32)
    Y_t2 = np.zeros((len(X),2,),dtype=np.float32)
    for i in range(len(Y)):
        if Y[i]=="FAKE":
            Y_t2[i,1] = 1.0
        else:
            Y_t2[i,0] = 1.0
    return X_t2, Y_t2

def get_dataset_2t():
    """Generate the data for the second dataset

    Returns:
        np.array,np.array: X and Y
    """
    df = pd.read_csv("data/news.csv")
    df = df.dropna()
    X = df["title"]
    Y = df["label"]
    with ProcessPoolExecutor() as executor:
        X_t2 = np.array(list(executor.map(to_npy_2,X)),dtype=np.float32)
    Y_t2 = np.zeros((len(X),2,),dtype=np.float32)
    for i in range(len(Y)):
        if Y[i]=="FAKE":
            Y_t2[i,1] = 1.0
        else:
            Y_t2[i,0] = 1.0
    return X_t2, Y_t2

def get_dataset_3():
    df = pd.read_csv("data/train.csv")
    df = df.dropna()
    X = df["text"]
    Y = list(df["label"])[:]
    with ProcessPoolExecutor() as executor:
        X_t2 = np.array(list(executor.map(to_npy,X)),dtype=np.float32)
    Y_t2 = np.zeros((len(X),2,),dtype=np.float32)
    for i in range(len(Y)):
        if Y[i] == 1:
            Y_t2[i,1] = 1.0
        else:
            Y_t2[i,0] = 1.0
    return X_t2, Y_t2
# arr = np.empty((48,-1,300,))
# for i in range(48):
#     arr[i] = x_real[i]

# https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes


