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
# [ ] Add PDP chart
# TODO:
# [ ] One hot encode
# [ ] Allow csv upload
# [ ] refactor
# [ ] Filter for misclassification
# GOOD-TO-HAVE:
# [ ] Display html formatted interpretation (ELI5/SHAP) (pending streamlit feature upgrade)
# [ ] Add other data types: text, image


# Title and Subheader
st.title("ML Interpretor")
st.subheader("Interpret model output with ELI5")


def splitdata(data, targetcol):
    """split dataset into trianing & testing"""
    X = data.drop(targetcol, axis=1)
    features = X.columns
    target_labels = data[targetcol].unique()
    y = pd.factorize(data[targetcol])[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80)
    return X_train, X_test, y_train, y_test, features, target_labels


@st.cache(suppress_st_warning=True)
def drawpdp(model, dataset, features, selected_feature):
    """draw pdpplot given a model, data, all the features and the selected feature to plot"""
    pdp_dist = pdp.pdp_isolate(model=model, dataset=dataset, model_features=features,
                               feature=selected_feature)
    pdp.pdp_plot(pdp_dist, selected_feature, ncols=3, plot_lines=True)
    st.pyplot()


def main():

    data_dim = st.sidebar.selectbox(
        'Try out sample data', ('iris', '20 news group'))

    ################################################
    # file upload
    ################################################

    uploaded_file = st.sidebar.file_uploader(
        "Or choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.text('upload complete')

    st.sidebar.selectbox('Then choose the target variable', [
                         'species', 'demo2'])

    model_dim = st.sidebar.selectbox(
        'Choose a model', ('randomforest',))

    ################################################
    # Model output
    ################################################
    if data_dim == 'iris':
        df = sns.load_dataset('iris')
        X_train, X_test, y_train, y_test, features, target_labels = splitdata(
            df, 'species')
        rf = RandomForestClassifier(n_estimators=500)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True)

    ################################################
    # Model output
    ################################################
    if st.sidebar.checkbox('Show classification report'):
        st.sidebar.dataframe(pd.DataFrame(report).transpose())

    ################################################
    # Global Interpretation
    ################################################

    # ELI5
    # As streamlit currently doesn't support the display of large chunks of HTML, the result below is mostly shown in tabular format
    st.markdown("#### Global Interpretation")
    st.text("Top feature importance")
    weights = pd.read_html(eli5.show_weights(
        rf, feature_names=features.values).data)
    st.dataframe(weights[0])

    ################################################
    # PDP plot
    ################################################
    st.markdown("#### Partial Dependence Plot")
    col = st.selectbox('Select feature', features)
    drawpdp(rf, X_train, features, col)

    ################################################
    # Local Interpretation
    ################################################
    st.markdown("#### Local Interpretation")
    # filter
    n_data = X_test.shape[0]
    slider_idx = st.slider("Which datapoint to explain", 0, n_data-1)
    # display input and prediction
    st.text('data to predict')
    st.dataframe(X_test.iloc[1, :])
    pred_label = pred[slider_idx]
    st.text('prediction: ' + target_labels[pred_label])
    local_interpretation = eli5.formatters.as_dataframe.explain_prediction_df(
        rf, X_train.iloc[slider_idx, :])
    local_interpretation_filtered = local_interpretation[local_interpretation.target == pred_label]
    st.dataframe(local_interpretation_filtered)


if __name__ == "__main__":
    main()
