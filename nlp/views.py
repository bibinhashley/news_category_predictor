from django.http.response import HttpResponse
from django.shortcuts import render
import os
from nlp.apps import NlpConfig
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
tfnews=pd.read_csv(path)


def home(request):    
    return render(request, 'nlp/index.html')


def result(request):
    news = request.GET["news"]
    num_similar_items=6
    
    tfidftext=vectorizer.fit_transform(tfnews['text'])
    tfidfnews=vectorizer.transform([news])
    couple_dist = pairwise_distances(tfidftext,tfidfnews)
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    dataframe=None
    dataframe=pd.DataFrame({'Recommended':tfnews['text'][indices].values[0:100]})
    recommended = dataframe.to_html()
    prediction = model.predict([news])
    return render(request, 'nlp/nlp.html', {'result':prediction[0],
                                            'recommendation':recommended})
    