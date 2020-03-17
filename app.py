import streamlit as st
import pandas as pd
# ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import lightgbm as lgb
import xgboost as xgb
from xgboost import DMatrix
# plotting
from bokeh.models.widgets import Div
import matplotlib.pyplot as plt
import seaborn as sns
# interpretation
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp


# DONE:
# [🎉] Sample tabular data
# [🎉] Display global and local interpretation
# [🎉] Add PDP chart
# [🎉] Allow csv upload
# [🎉] auto encode
# [🎉] Filter for misclassification
# [🎉] deploy to heroku
# [🎉] add more ml algos: xgb, lgbm
# TODO:
# [ ] add another demo data
# [ ] add pdp for xgb
# [ ] add distribution plot for individual datapoint
# [ ] add circleCI
# GOOD-TO-HAVE:
# [ ] Add shields.io
# [ ] Allow model upload
# [ ] add other interpretation framework (SHAP, LIME etc)
# [ ] add two variable interaction pdp (pending pdpbox maintainer fix)
# [ ] Add other data types: text, image


# Title and Subheader
st.title("ML Interpretor")
st.subheader("Interpret model output with ELI5")


def splitdata(data, targetcol):
    """preprocess categorical value and split dataset into trianing & testing"""
    X = pd.get_dummies(data.drop(targetcol, axis=1)).fillna(0)
    features = X.columns
    data[targetcol] = data[targetcol].astype('object')
    target_labels = data[targetcol].unique()
    y = pd.factorize(data[targetcol])[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=0)
    return X_train, X_test, y_train, y_test, features, target_labels


def drawpdp(model, dataset, features, selected_feature, target_labels, model_dim):
    """draw pdpplot given a model, data, all the features and the selected feature to plot"""
    if model_dim != 'XGBoost':
        pdp_dist = pdp.pdp_isolate(model=model, dataset=dataset, model_features=features,
                                   feature=selected_feature)
        if len(target_labels) <= 5:
            ncol = len(target_labels)
        else:
            ncol = 5
        pdp.pdp_plot(pdp_dist, selected_feature, ncols=ncol, plot_lines=True)
        st.pyplot()


def show_local_interpretation(dataset, clf, pred, target_labels, features, model_dim):
    """show individual decision points"""
    info_local = st.button('How this works')
    if info_local:
        st.markdown("""
        #### What's included
        Input data is split 80/20 into training and testing. Model is trained on the training and applied on the testing.
        Each of the individual testing datapoint can be inspected by index. 
        #### To Read the table
        The table describes how an individual datapoint is classified.
        Contribution refers to the extent & direction of influence a feature has on the outcome 
        Value refers to the value of the feature in the dataset
        """)
    n_data = dataset.shape[0]
    slider_idx = st.slider("Which datapoint to explain", 0, n_data-1)

    pred_label = pred[slider_idx]
    st.text('prediction: ' + str(target_labels[int(pred_label)]))
    if model_dim != 'XGBoost':
        local_interpretation = eli5.show_prediction(
            clf, doc=dataset.iloc[slider_idx, :], target_names=target_labels, show_feature_values=True, top=5, targets=[True])
    else:
        local_interpretation = eli5.show_prediction(
            clf, doc=dataset.iloc[slider_idx, :], show_feature_values=True, top=5)
    st.markdown(local_interpretation.data.translate(
        str.maketrans('', '', '\n')), unsafe_allow_html=True)


def main():

    data_dim = st.sidebar.selectbox(
        'Try out sample data', ('iris', ''))

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
        'Choose a model', ('randomforest', 'lightGBM', 'XGBoost'))
    if model_dim == 'randomforest':
        clf = RandomForestClassifier(n_estimators=500, random_state=0)
        clf.fit(X_train, y_train)
    elif model_dim == 'lightGBM':
        if len(target_labels) > 2:
            clf = lgb.LGBMClassifier(
                class_weight='balanced',
                objective='multiclass',
                n_jobs=-1,
                verbose=0)
        else:
            clf = lgb.LGBMClassifier(
                objective='binary',
                n_jobs=-1,
                verbose=0)
        clf.fit(X_train, y_train)
    elif model_dim == 'XGBoost':
        params = {'max_depth': 5, 'silent': 1,
                  'random_state': 2, 'num_class': len(target_labels)}
        dmatrix = DMatrix(data=X_train, label=y_train)
        clf = xgb.train(params=params, dtrain=dmatrix)

    ################################################
    # evaluate prediction
    ################################################
    if model_dim == 'XGBoost':
        pred = clf.predict(DMatrix(X_test))
    else:
        pred = clf.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)
    X_test_misclassified = X_test[pred != y_test]

    ################################################
    # Model output
    ################################################
    if st.sidebar.checkbox('Preview uploaded data'):
        st.sidebar.dataframe(df.head())

    # the report is formatted to 2 decimal points (i.e. accuracy 1 means 1.00) dependent on streamlit styling update https://github.com/streamlit/streamlit/issues/1125
    if st.sidebar.checkbox('Show classification report'):
        st.sidebar.table(pd.DataFrame(report).round(2).transpose())
    # TODO: add at least 1 decimal rounding
    if st.sidebar.button('About the app'):
        st.sidebar.markdown(
            """  
             Read more about how it works on [Github] (https://github.com/yanhann10/ml_interpret)   
             Last update Mar 2020.
             Contact @hannahyan.  
              """)

    ################################################
    # Global Interpretation
    ################################################

    # ELI5
    st.markdown("#### Global Interpretation")
    st.text("Top feature importance")
    info_global = st.button('How it is calculated')
    if info_global:
        st.markdown("""
        #### How this works
        The importance of each feature is derived from [permutation importance](https://www.kaggle.com/dansbecker/permutation-importance) -
        by randomly shuffle a feature, how much does the model performance decrease. The ± takes into account the randomness of shuffles.
        """)
    # This only works if removing newline from html
    # Refactor this once added more models
    if model_dim == 'randomforest':
        global_interpretation = eli5.show_weights(
            clf, feature_names=features.values, top=5).data
    elif model_dim == 'lightGBM':
        perm = PermutationImportance(
            clf, random_state=1).fit(X_train, y_train)
        global_interpretation = eli5.show_weights(
            perm, feature_names=X_train.columns.tolist(), top=5).data
    elif model_dim == 'XGBoost':
        global_interpretation = eli5.show_weights(
            clf,  top=5).data

    st.markdown(global_interpretation.translate(
        str.maketrans('', '', '\n')), unsafe_allow_html=True)

    ################################################
    # PDP plot
    ################################################
    st.markdown("#### How features relate to outcome")
    select_col = st.selectbox('Select a feature', features)

    url_pdp = 'https://christophm.github.io/interpretable-ml-book/pdp.html'
    info_pdp = st.button('How to read the chart')
    if info_pdp:
        st.markdown("""The curves describe how a feature varies with the likelihood of outcome. Each subplot belong to a class outcome.
        When a curve is below 0, the data is unlikely to belong to that class.
        [Read more] (url_pdp) """)

    drawpdp(clf, X_train, features, select_col, target_labels, model_dim)

    ################################################
    # Local Interpretation
    ################################################
    st.markdown("#### Local Interpretation")

    if st.checkbox('Filter for misclassified'):
        if X_test_misclassified.shape[0] == 0:
            st.text('No misclassification🎉')
        else:
            show_local_interpretation(
                X_test_misclassified, clf, pred, target_labels, features, model_dim)
    else:
        show_local_interpretation(
            X_test, clf, pred, target_labels, features, model_dim)


if __name__ == "__main__":
    main()
