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
st.subheader("Interpret model output with ELI5")

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

if st.checkbox('Show prediction Outcome'):
    st.text(f'Metric {metric_accuracy:.3f}')


# Interpretation
if st.checkbox("Global Interpretation"):
    weights = pd.read_html(eli5.show_weights(rf).data)
    st.dataframe(weights[0])

if st.checkbox("Local Interpretation"):
    n_data = train.shape[0]
    slider_data = st.slider("Which datapoint to explain", 0, n_data)
    local_interpretation = eli5.formatters.as_dataframe.explain_prediction_df(
        rf, train[slider_data])
    st.dataframe(local_interpretation)


# explainer = lime.lime_tabular.LimeTabularExplainer(
#     train, feature_names=data.feature_names, class_names=data.target_names, discretize_continuous=True)

# i = np.random.randint(0, test.shape[0])
# exp = explainer.explain_instance(
#     test[i], rf.predict_proba, num_features=2, top_labels=1)

# exp.show_in_notebook(show_table=True, show_all=False)
