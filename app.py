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
import shap
import eli5
from PIL import Image, ImageFilter, ImageEnhance
from pdpbox import pdp, get_dataset, info_plots
# pipeline
import joblib

# DONE:
# [🎉] Sample tabular data
# [🎉] Display global and local interpretation
# [🎉] Allow csv upload
# TODO:
# [ ] Add PDP chart
# [ ] Display properly formatted interp
# [ ] Filter for wrong classification
# [ ] Display proper feature
# [ ] Add other data types text, image
# [ ] Add SHAP


# Title and Subheader
st.title("ML Interpretor")
st.subheader("Interpret model output with ELI5")


def main():

    ################################################
    # Side bar control
    ################################################

    data_dim = st.sidebar.radio(
        'Try out sample data', ('iris', ))

    filename = st.sidebar.text_input('Or enter a csv file path:')
    if filename != "":
        try:
            with open(filename) as input:
                df = input.read()
                st.sidebar.text('upload complete')
        except FileNotFoundError:
            st.sidebar.error('File not found.')

    model_dim = st.sidebar.radio(
        'Choose a model', ('randomforest',))

    ################################################
    # Model output
    ################################################
    if data_dim == 'iris':
        iris = sns.load_dataset('iris')
        X = iris.drop("species", axis=1)
        y_labels = iris.species.unique()
        y = pd.factorize(iris['species'])[0]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.80)
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True)
    # elif data_dim == '20newsgroup':
    #     categories = ['comp.graphics', 'sci.med']
    #     twenty_train = sklearn.datasets.fetch_20newsgroups(subset='train',
    #                                                        categories=categories, shuffle=True, random_state=42
    #                                                        )
    #     twenty_test = sklearn.datasets.fetch_20newsgroups(
    #         subset='test', categories=categories, shuffle=True, random_state=42
    #     )
    #     vec = CountVectorizer()
    #     rf = RandomForestClassifier()
    #     clf = make_pipeline(vec, rf)
    #     clf.fit(twenty_train.data, twenty_train.target)
    #     pred = clf.predict(twenty_test.data)
    #     report = classification_report(
    #         twenty_test.target, pred, output_dict=True)

    ################################################
    # Model output
    ################################################
    if st.sidebar.checkbox('Show classification report'):
        st.sidebar.dataframe(pd.DataFrame(report).transpose())

    ################################################
    # Interpretation
    ################################################

    # ELI5
    # due to streamlit currently doesn't support the display of large chunks of HTML, the result below is mostly shown in tabular format
    st.markdown("#### Global Interpretation")
    weights = pd.read_html(eli5.show_weights(
        rf, feature_names=X.columns.values).data)
    st.dataframe(weights[0])

    st.markdown("#### Local Interpretation")
    # filter
    n_data = X_test.shape[0]
    slider_idx = st.slider("Which datapoint to explain", 0, n_data-1)
    # display input and prediction
    st.text('input')
    st.dataframe(X_test.iloc[1, :])
    pred_label = pred[slider_idx]
    st.text('prediction: ' + y_labels[pred_label])
    local_interpretation = eli5.formatters.as_dataframe.explain_prediction_df(
        rf, X_train.iloc[slider_idx, :])
    local_interpretation_filtered = local_interpretation[local_interpretation.target == pred_label]
    st.dataframe(local_interpretation_filtered)

    st.markdown("#### Partial Dependence Plot")
    pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_train, model_features=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                               feature='sepal_length')
    pdp.pdp_plot(pdp_dist, 'sepal_length')
    st.pyplot()


if __name__ == "__main__":
    main()
