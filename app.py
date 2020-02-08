import streamlit as st
import pandas as pd
import numpy as np
import os
# ml
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# interpretation
import lime
import eli5
from PIL import Image, ImageFilter, ImageEnhance
# pipeline
import joblib

### Title and Subheader
st.title("ML Interpretor")
st.subheader("Understand why a model is generating certain results")

# File input
# TODO File upload
# TODO url upload
# TODO model upload
# datafile_vectorizer = open("mdl.pkl",'rb')
# datafile= joblib.load(mdl_vectorizer)
# mdl_vectorizer = open("mdl.pkl" ,'rb')
# mdl= joblib.load(mdl_vectorizer)
st.sidebar.text("Upload a csv")

# Side bar settings
# TODO Sample data: Tabular, text, image
data_dim = st.sidebar.radio('Try out sample data', ('iris', '20newsgroup'))
if data_dim == 'iris':
    data = sklearn.datasets.load_iris()

data_dim = st.sidebar.radio('Choose a model', ('randomforest', 'catBoost'))

st.sidebar.button("About App")

# Model
train, test, labels_train, labels_test = train_test_split(
    data.data, data.target, train_size=0.80)

rf = RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)

metric_accuracy = sklearn.metrics.accuracy_score(labels_test, rf.predict(test))

st.text(f'Metric {metric_accuracy}')

# Model when uploaded
# def model_predict(data):
#     vect = data_csv.transform(data).toarray()
#     result = mdl.predict(vect)
#     return result


# Interpretation
if st.checkbox("Global Interpretation"):
    st.text("Global Interpretation")


st.text("Local Interpretation")

weights = pd.read_html(eli5.show_weights(rf).data)
st.dataframe(weights[0])

html_temp = """
<div style="color:tomato;"> i'm html</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# eli5.show_prediction(rf, train[1, :], show_feature_values=True)


# explainer = lime.lime_tabular.LimeTabularExplainer(
#     train, feature_names=data.feature_names, class_names=data.target_names, discretize_continuous=True)

# i = np.random.randint(0, test.shape[0])
# exp = explainer.explain_instance(
#     test[i], rf.predict_proba, num_features=2, top_labels=1)

# exp.show_in_notebook(show_table=True, show_all=False)

