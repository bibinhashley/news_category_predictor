import os
import joblib
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(__file__)
model_file = os.path.join(CURRENT_DIR, 'model.file')
vectorizer=os.path.join(CURRENT_DIR, 'tfidfvectorizer.file')
model = joblib.load(model_file)
vectorizer=joblib.load(vectorizer)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, 'nlp/models/bbc-text.csv')
fnews=pd.read_csv(path)['text']

print(fnews)