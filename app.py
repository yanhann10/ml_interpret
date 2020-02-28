import streamlit as st
import pandas as pd
import numpy as np
# ml
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# interpretation
import lime
import eli5
from PIL import Image, ImageFilter, ImageEnhance
# pipeline
import joblib

# TODO:
# [ ] Allow csv upload #st.sidebar.text("Upload a csv")
# [ ] Add other data types text, image
# [ ] Support H2O autoML
# [ ] Add LIME, SHAP
# [ ] Add chart

# DONE:
# [🎉] Tabular data
# [🎉] display global and local interpretation

################################################
# side bar
################################################

### Title and Subheader
st.title("ML Interpretor")
st.subheader("Interpret model output with ELI5")


# datafile_vectorizer = open("mdl.pkl",'rb')
# datafile= joblib.load(mdl_vectorizer)
# mdl_vectorizer = open("mdl.pkl" ,'rb')
# mdl= joblib.load(mdl_vectorizer)


################################################
# Model output
################################################

data_dim = st.sidebar.radio('Try out sample data', ('iris', '20newsgroup'))
if data_dim == 'iris':
    data = sklearn.datasets.load_iris()
    train, test, labels_train, labels_test = train_test_split(
        data.data, data.target, train_size=0.80)
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(train, labels_train)
    pred = rf.predict(test)
    report = classification_report(labels_test, pred, output_dict=True)
elif data_dim == '20newsgroup':
    categories = ['comp.graphics', 'sci.med']
    twenty_train = sklearn.datasets.fetch_20newsgroups(subset='train',
                                                       categories=categories, shuffle=True, random_state=42
                                                       )
    twenty_test = sklearn.datasets.fetch_20newsgroups(
        subset='test', categories=categories, shuffle=True, random_state=42
    )
    vec = CountVectorizer()
    rf = RandomForestClassifier()
    clf = make_pipeline(vec, rf)
    clf.fit(twenty_train.data, twenty_train.target)
    pred = clf.predict(twenty_test.data)
    report = classification_report(twenty_test.target, pred, output_dict=True)

data_dim = st.sidebar.radio('Choose a model', ('randomforest', 'catBoost'))

################################################
# Model output
################################################

if st.checkbox('Show prediction Outcome'):
    st.dataframe(pd.DataFrame(report).transpose())

################################################
# Interpretation
################################################
if st.checkbox("Global Interpretation"):
    weights = pd.read_html(eli5.show_weights(rf).data)
    st.dataframe(weights[0])

if st.checkbox("Local Interpretation"):
    n_data = train.shape[0]
    slider_data = st.slider("Which datapoint to explain", 0, n_data)
    local_interpretation = eli5.formatters.as_dataframe.explain_prediction_df(
        rf, train[slider_data])
    st.dataframe(local_interpretation)
    explaination = eli5.explain_prediction(rf, train[slider_data])
    st.image(eli5.formatters.image.format_as_image(explaination))


################################################
# About
################################################
if st.button("About App"):
    st.write(
        "This app demostrates machine learning interpretation frameworks such as ELI on sample data and user-chosen algorithms")
