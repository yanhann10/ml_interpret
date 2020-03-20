import streamlit as st
import pandas as pd
# ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from xgboost import DMatrix
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
# interpretation
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp
import shap
shap.initjs()


# DONE:
# [🎉] Sample tabular data
# [🎉] Display global and local interpretation
# [🎉] Add PDP chart
# [🎉] Allow csv upload
# [🎉] auto encode
# [🎉] Filter for misclassification
# [🎉] deploy to heroku
# [🎉] add more ml algos: xgb, lgbm
# [🎉] add confusion matrix
# [🎉] add other interpretation framework (SHAP etc)
# TODO:
# [ ] deal with permutation impt speed issue
# [ ] add another demo data
# [ ] add pdp for xgb
# GOOD-TO-HAVE:
# [ ] add distribution plot for individual datapoint
# [ ] add plot for tree surrogate
# [ ] Allow model upload
# [ ] add two variable interaction pdp (pending pdpbox maintainer fix)
# [ ] Add other data types: text, image


# Title and Subheader
st.title("ML Interpretor")
st.subheader("Interpret blackblox ML models")


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


@st.cache(suppress_st_warning=True)
def draw_pdp(model, dataset, features, target_labels, model_dim):
    """draw pdpplot given a model, data, all the features and the selected feature to plot"""

    if model_dim != 'XGBoost':
        st.markdown("#### How features relate to outcome")
        selected_col = st.selectbox('Select a feature', features)

        url_pdp = 'https://christophm.github.io/interpretable-ml-book/pdp.html'
        info_pdp = st.button('How to read the chart')
        if info_pdp:
            st.info("""The curves describe how a feature varies with the likelihood of outcome. Each subplot belong to a class outcome.
            When a curve is below 0, the data is unlikely to belong to that class.
            [Read more] (url_pdp) """)

        pdp_dist = pdp.pdp_isolate(model=model, dataset=dataset, model_features=features,
                                   feature=selected_col)
        if len(target_labels) <= 5:
            ncol = len(target_labels)
        else:
            ncol = 5
        pdp.pdp_plot(pdp_dist, selected_col, ncols=ncol, plot_lines=True)
        st.pyplot()


def show_global_interpretation_eli5(X_train, y_train, features, clf, model_dim):
    """show most important features via permutation importance"""
    if model_dim == 'XGBoost':
        df_global_explain = eli5.explain_weights_df(
            clf, feature_names=X_train.columns.values, top=5)
    else:
        perm = PermutationImportance(
            clf, random_state=1).fit(X_train, y_train)
        df_global_explain = eli5.explain_weights_df(
            perm, feature_names=X_train.columns.values, top=5)
    df_global_explain = df_global_explain.round(2)
    bar = alt.Chart(df_global_explain).mark_bar(opacity=0.6).encode(
        x='weight',
        y=alt.Y('feature', sort='-x'),
        tooltip=['weight']
    ).properties(height=100)

    st.write(bar)


def show_global_interpretation_shap(X_train, clf):
    """global interpretation with SHAP"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train,
                      plot_type="bar", max_display=5, plot_size=(14, 4))
    st.pyplot()


def show_local_interpretation_eli5(dataset, clf, pred, target_labels, features, model_dim, slider_idx):
    """show the interpretation of individual decision points"""
    info_local = st.button('How this works')
    if info_local:
        st.info("""
        ## What's included
        Input data is split 80/20 into training and testing. Model is trained on the training and applied on the testing.
        Each of the individual testing datapoint can be inspected by index. 
        ## To Read the table
        The table describes how an individual datapoint is classified.
        Contribution refers to the extent & direction of influence a feature has on the outcome 
        Value refers to the value of the feature in the dataset
        """)

    pred_label = pred[slider_idx]
    # st.text('prediction: ' + str(target_labels[int(pred_label)]))
    if model_dim == 'XGBoost':
        local_interpretation = eli5.show_prediction(
            clf, doc=dataset.iloc[slider_idx, :], show_feature_values=True, top=5)
    else:
        local_interpretation = eli5.show_prediction(
            clf, doc=dataset.iloc[slider_idx, :], target_names=target_labels, show_feature_values=True, top=5, targets=[True])
    st.markdown(local_interpretation.data.translate(
        str.maketrans('', '', '\n')), unsafe_allow_html=True)


def show_local_interpretation_shap(clf, X_test, slider_idx):
    """show the interpretation of individual decision points"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    shap.force_plot(
        explainer.expected_value[slider_idx], shap_values[1][slider_idx, :], X_test.iloc[slider_idx, :], matplotlib=True)
    st.pyplot()


def show_perf_metrics(y_test, pred):
    """show model performance metrics such as classification report or confusion matrix"""
    report = classification_report(y_test, pred, output_dict=True)
    st.sidebar.dataframe(pd.DataFrame(report).round(1).transpose())
    conf_matrix = pd.DataFrame(confusion_matrix(
        y_test, pred, list(set(y_test))))
    sns.set_style("dark")
    sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", cbar=False)
    st.sidebar.pyplot()

    alt.Chart(conf_matrix).mark_rect().encode(
        x='y_test',
        y='pred',
        color='z:Q'
    )


def main():
    ################################################
    # upload file
    ################################################
    data_dim = st.sidebar.selectbox(
        'Try out sample data', ('iris', ''))
    uploaded_file = st.sidebar.file_uploader(
        "Or upload a CSV file", type="csv")

    target_col = ''
    if uploaded_file is not None:
        st.sidebar.success('File uploaded!')
        df = pd.read_csv(uploaded_file)
        # make the last col the default outcome
        col_arranged = df.columns[:-1].insert(0, df.columns[-1])
        target_col = st.sidebar.selectbox(
            'Then choose the target variable', col_arranged)
    elif data_dim == 'iris':
        df = sns.load_dataset('iris')
        target_col = 'species'

    ################################################
    # process data
    ################################################

    X_train, X_test, y_train, y_test, features, target_labels = splitdata(
        df, target_col)

    ################################################
    # apply model
    ################################################
    model_dim = st.sidebar.selectbox(
        'Choose a model', ('lightGBM', 'XGBoost', 'randomforest'))
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
    # Predict
    ################################################
    if model_dim == 'XGBoost':
        pred = clf.predict(DMatrix(X_test))
    else:
        pred = clf.predict(X_test)

    X_test_misclassified = X_test[pred != y_test]
    pred_misclassified = pred[pred != y_test]
    #y_test_misclassified = y_test[pred != y_test]

    ################################################
    # Model output
    ################################################
    if st.sidebar.checkbox('Preview uploaded data'):
        st.sidebar.dataframe(df.head())

    # the report is formatted to 2 decimal points (i.e. accuracy 1 means 1.00) dependent on streamlit styling update https://github.com/streamlit/streamlit/issues/1125
    if st.sidebar.checkbox('Show classification report'):
        show_perf_metrics(y_test, pred)

    dim_framework = st.sidebar.radio(
        "Choose interpretation framework", ['SHAP', 'ELI5'])

    if st.sidebar.button('About the app'):
        st.sidebar.markdown(
            """  
             Read more about how it works on [Github] (https://github.com/yanhann10/ml_interpret)     
             Last update Mar 2020  
             [Feedback](https://docs.google.com/forms/d/e/1FAIpQLSdTXKpMPC0-TmWf2ngU9A0sokH5Z0m-QazSPBIZyZ2AbXIBug/viewform?usp=sf_link)  
             Contact @hannahyan.  
              """)

    ################################################
    # Global Interpretation
    ################################################

    st.markdown("#### Global Interpretation")
    st.text("Most important features")
    info_global = st.button('How it is calculated')
    if info_global:
        st.info("""
        The importance of each feature is derived from [permutation importance](https://www.kaggle.com/dansbecker/permutation-importance) -
        by randomly shuffle a feature, how much does the model performance decrease. 
        """)
    # This only works if removing newline from html
    # Refactor this once added more models
    if dim_framework == 'SHAP':
        show_global_interpretation_shap(X_train, clf)
    elif dim_framework == 'ELI5':
        show_global_interpretation_eli5(
            X_train, y_train, features, clf, model_dim)

    ################################################
    # PDP plot
    ################################################

    #draw_pdp(clf, X_train, features, target_labels, model_dim)

    ################################################
    # Local Interpretation
    ################################################
    st.markdown("#### Local Interpretation")
    n_data = X_test.shape[0]
    slider_idx = st.slider("Which datapoint to explain", 0, n_data-1)

    if dim_framework == 'SHAP':
        show_local_interpretation_shap(clf, X_test, slider_idx)
    elif dim_framework == 'ELI5':
        show_local_interpretation_eli5(
            X_test, clf, pred, target_labels, features, model_dim, slider_idx)

    # if st.checkbox('Filter for misclassified'):
    #     if X_test_misclassified.shape[0] == 0:
    #         st.text('No misclassification🎉')
    #     else:
    #         show_local_interpretation(
    #             X_test_misclassified, clf, pred_misclassified, target_labels, features, model_dim, slider_idx)
    # else:
    #     show_local_interpretation(
    #         X_test, clf, pred, target_labels, features, model_dim, slider_idx)


if __name__ == "__main__":
    main()
