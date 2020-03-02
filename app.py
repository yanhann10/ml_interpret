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
import eli5
from pdpbox import pdp, get_dataset, info_plots
# pipeline
import joblib

# DONE:
# [🎉] Sample tabular data
# [🎉] Display global and local interpretation
# [🎉] Add PDP chart
# [🎉 ] Allow csv upload
# [🎉] auto encode
# TODO:
# [ ] look into outcome categorical
# [ ] add ml algos
# [ ] connect encoded label n pdp
# [ ] Filter for misclassification
# [ ] Allow model upload
# [ ] deploy to GCP or heroku
# GOOD-TO-HAVE:
# [ ] add circleCI
# [ ] Display html formatted interpretation (ELI5/SHAP) (pending streamlit feature upgrade)
# [ ] Add other data types: text, image


# Title and Subheader
st.title("ML Interpretor")
st.subheader("Interpret model output with ELI5")


def splitdata(data, targetcol):
    """preprocess categorical value and split dataset into trianing & testing"""
    cols = data.columns
    X = pd.get_dummies(data.drop(targetcol, axis=1)).fillna(0)
    features = X.columns
    data[targetcol] = data[targetcol].astype('object')
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
    # upload file
    ################################################

    uploaded_file = st.sidebar.file_uploader(
        "Or choose a CSV file", type="csv")

    target_col = ''
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.text('upload complete')
        target_col = st.sidebar.selectbox(
            'Then choose the target variable', df.columns)

    ################################################
    # process data
    ################################################

    elif data_dim == 'iris':
        df = sns.load_dataset('iris')
        target_col = 'species'

    X_train, X_test, y_train, y_test, features, target_labels = splitdata(
        df, target_col)

    ################################################
    # apply model
    ################################################
    model_dim = st.sidebar.selectbox(
        'Choose a model', ('randomforest', ))
    if model_dim == 'randomforest':
        clf = RandomForestClassifier(n_estimators=500, random_state=0)
        clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)

    ################################################
    # Model output
    ################################################
    if st.sidebar.checkbox('Preview data'):
        st.sidebar.dataframe(df.head())

    if st.sidebar.checkbox('Show classification report'):
        st.sidebar.dataframe(pd.DataFrame(report).transpose())

    if st.sidebar.button('About the app'):
        st.sidebar.markdown(
            """
             Last update Mar 2020.    
             [Github] (https://github.com/yanhann10/ml_interpret)   
             Contact @hannahyan.  
              """)

    ################################################
    # Global Interpretation
    ################################################

    # ELI5
    # As streamlit currently doesn't support the display of large chunks of HTML, the result below is mostly shown in tabular format
    st.markdown("#### Global Interpretation")
    st.text("Top feature importance")
    weights = pd.read_html(eli5.show_weights(
        clf, feature_names=features.values).data)
    st.dataframe(weights[0])

    ################################################
    # PDP plot
    ################################################
    st.markdown("#### Partial Dependence Plot")
    col = st.selectbox('Select feature', features)
    drawpdp(clf, X_train, features, col)

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
    st.text('prediction: ' + str(target_labels[pred_label]))
    local_interpretation = eli5.formatters.as_dataframe.explain_prediction_df(
        clf, X_train.iloc[slider_idx, :])
    local_interpretation_filtered = local_interpretation[local_interpretation.target == pred_label]
    st.dataframe(local_interpretation_filtered)


if __name__ == "__main__":
    main()
