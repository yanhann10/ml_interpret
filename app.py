import streamlit as st
import pandas as pd

# ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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

# Title and Subheader
st.title("ML Interpreter")
st.subheader("Blackblox ML classifiers visually explained")


def upload_data(uploaded_file, dim_data):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded!")
        df = pd.read_csv(uploaded_file, encoding="utf8")
        # replace all non alphanumeric column names to avoid lgbm issue
        df.columns = [
            "".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns
        ]
        # make the last col the default outcome
        col_arranged = df.columns[:-1].insert(0, df.columns[-1])
        target_col = st.sidebar.selectbox(
            "Then choose the target variable", col_arranged
        )
        X, y, features, target_labels = encode_data(df, target_col)
    elif dim_data == "iris":
        df = sns.load_dataset("iris")
        target_col = "species"
        X, y, features, target_labels = encode_data(df, target_col)
    elif dim_data == "titanic":
        df = sns.load_dataset("titanic").drop(
            columns=["class", "who", "adult_male", "deck", "alive", "alone"]
        )
        target_col = "survived"
        X, y, features, target_labels = encode_data(df, target_col)
    elif dim_data == "census income":
        X, y = shap.datasets.adult()
        features = X.columns
        target_labels = pd.Series(y).unique()
        df = pd.concat([X, pd.DataFrame(y, columns=["Outcome"])], axis=1)
    return df, X, y, features, target_labels


def encode_data(data, targetcol):
    """preprocess categorical value"""
    X = pd.get_dummies(data.drop(targetcol, axis=1)).fillna(0)
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    features = X.columns
    data[targetcol] = data[targetcol].astype("object")
    target_labels = data[targetcol].unique()
    y = pd.factorize(data[targetcol])[0]
    return X, y, features, target_labels


def splitdata(X, y):
    """split dataset into trianing & testing"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=0
    )
    return X_train, X_test, y_train, y_test


def make_pred(dim_model, X_test, clf):
    """get y_pred using the classifier"""
    if dim_model == "XGBoost":
        pred = clf.predict(DMatrix(X_test))
    elif dim_model == "lightGBM":
        pred = clf.predict(X_test)
    else:
        pred = clf.predict(X_test)
    return pred


def show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model):
    """show most important features via permutation importance in ELI5"""
    if dim_model == "XGBoost":
        df_global_explain = eli5.explain_weights_df(
            clf, feature_names=features.values, top=5
        ).round(2)
    else:
        perm = PermutationImportance(clf, n_iter=2, random_state=1).fit(
            X_train, y_train
        )
        df_global_explain = eli5.explain_weights_df(
            perm, feature_names=features.values, top=5
        ).round(2)
    bar = (
        alt.Chart(df_global_explain)
        .mark_bar(color="red", opacity=0.6, size=16)
        .encode(x="weight", y=alt.Y("feature", sort="-x"), tooltip=["weight"])
        .properties(height=160)
    )
    st.write(bar)


def show_global_interpretation_shap(X_train, clf):
    """show most important features via permutation importance in SHAP"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(
        shap_values,
        X_train,
        plot_type="bar",
        max_display=5,
        plot_size=(12, 5),
        color=plt.get_cmap("tab20b"),
        show=False,
        color_bar=False,
    )
    # note: there might be figure cutoff issue. Will look further into forceplot & st.pyplot's implementation.
    st.pyplot()


def filter_misclassified(X_test, y_test, pred):
    """get misclassified instances"""
    idx_misclassified = pred != y_test
    X_test_misclassified = X_test[idx_misclassified]
    y_test_misclassified = y_test[idx_misclassified]
    pred_misclassified = pred[idx_misclassified]
    return X_test_misclassified, y_test_misclassified, pred_misclassified


def show_local_interpretation_eli5(
    dataset, clf, pred, target_labels, features, dim_model, slider_idx
):
    """show the interpretation of individual decision points"""
    info_local = st.button("How this works")
    if info_local:
        st.info(
            """
        **What's included**  
        Input data is split 80/20 into training and testing. 
        Each of the individual testing datapoint can be inspected by index.
        **To Read the table**  
        The table describes how an individual datapoint is classified.
        Contribution refers to the extent & direction of influence a feature has on the outcome
        Value refers to the value of the feature in the dataset. Bias means an intercept.
        """
        )

    if dim_model == "XGBoost":
        local_interpretation = eli5.show_prediction(
            clf, doc=dataset.iloc[slider_idx, :], show_feature_values=True, top=5
        )
    else:
        local_interpretation = eli5.show_prediction(
            clf,
            doc=dataset.iloc[slider_idx, :],
            target_names=target_labels,
            show_feature_values=True,
            top=5,
            targets=[True],
        )
    st.markdown(
        local_interpretation.data.replace("\n", ""), unsafe_allow_html=True,
    )


def show_local_interpretation_shap(clf, X_test, pred, slider_idx):
    """show the interpretation of individual decision points"""
    info_local = st.button("How this works")
    if info_local:
        st.info(
            """
        This chart illustrates how each feature collectively influence the prediction outcome.
        Features in the red make it more likely to be the predicted class, and the features in blue pushing back leftward reduce the likelihood. [Read more about forceplot](https://github.com/slundberg/shap)  
        Please note that the explanation here is always based on the predicted class rather than the positive class (i.e. if predicted class is 0, to the right means more likely to be 0) to cater for multi-class senaiors.
        """
        )
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    # the predicted class for the selected instance
    pred_i = int(pred[slider_idx])
    # this illustrates why the model predict this particular outcome
    shap.force_plot(
        explainer.expected_value[pred_i],
        shap_values[pred_i][slider_idx, :],
        X_test.iloc[slider_idx, :],
        matplotlib=True,
    )
    st.pyplot()


def show_local_interpretation(
    X_test, y_test, clf, pred, target_labels, features, dim_model, dim_framework
):
    """show the interpretation based on the selected framework"""
    n_data = X_test.shape[0]
    slider_idx = st.slider("Which datapoint to explain", 0, n_data - 1)

    st.text(
        "Prediction: "
        + str(target_labels[int(pred[slider_idx])])
        + " | Actual label: "
        + str(target_labels[int(y_test[slider_idx])])
    )

    if dim_framework == "SHAP":
        show_local_interpretation_shap(clf, X_test, pred, slider_idx)
    elif dim_framework == "ELI5":
        show_local_interpretation_eli5(
            X_test, clf, pred, target_labels, features, dim_model, slider_idx
        )


def show_perf_metrics(y_test, pred):
    """show model performance metrics such as classification report or confusion matrix"""
    report = classification_report(y_test, pred, output_dict=True)
    st.sidebar.dataframe(pd.DataFrame(report).round(1).transpose())
    conf_matrix = confusion_matrix(y_test, pred, list(set(y_test)))
    sns.set(font_scale=1.4)
    sns.heatmap(
        conf_matrix,
        square=True,
        annot=True,
        annot_kws={"size": 15},
        cmap="YlGnBu",
        cbar=False,
    )
    st.sidebar.pyplot()


def draw_pdp(clf, dataset, features, target_labels, dim_model):
    """draw pdpplot given a model, data, all the features and the selected feature to plot"""

    if dim_model != "XGBoost":
        selected_col = st.selectbox("Select a feature", features)
        st.info(
            """**To read the chart:** The curves describe how a feature marginally varies with the likelihood of outcome. Each subplot belong to a class outcome.
        When a curve is below 0, the data is unlikely to belong to that class.
        [Read more] ("https://christophm.github.io/interpretable-ml-book/pdp.html") """
        )

        pdp_dist = pdp.pdp_isolate(
            model=clf, dataset=dataset, model_features=features, feature=selected_col
        )
        if len(target_labels) <= 5:
            ncol = len(target_labels)
        else:
            ncol = 5
        pdp.pdp_plot(pdp_dist, selected_col, ncols=ncol, figsize=(12, 5))
        st.pyplot()


def main():
    ################################################
    # upload file
    ################################################
    dim_data = st.sidebar.selectbox(
        "Try out sample data", ("iris", "titanic", "census income")
    )
    uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type="csv")

    df, X, y, features, target_labels = upload_data(uploaded_file, dim_data)

    ################################################
    # process data
    ################################################

    X_train, X_test, y_train, y_test = splitdata(X, y)

    ################################################
    # apply model
    ################################################
    dim_model = st.sidebar.selectbox(
        "Choose a model", ("XGBoost", "lightGBM", "randomforest")
    )
    if dim_model == "randomforest":
        clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)
    elif dim_model == "lightGBM":
        if len(target_labels) > 2:
            clf = lgb.LGBMClassifier(
                class_weight="balanced", objective="multiclass", n_jobs=-1, verbose=-1
            )
        else:
            clf = lgb.LGBMClassifier(objective="binary", n_jobs=-1, verbose=-1)
        clf.fit(X_train, y_train)
    elif dim_model == "XGBoost":
        params = {
            "max_depth": 5,
            "silent": 1,
            "random_state": 2,
            "num_class": len(target_labels),
        }
        dmatrix = DMatrix(data=X_train, label=y_train)
        clf = xgb.train(params=params, dtrain=dmatrix)

    ################################################
    # Predict
    ################################################
    pred = make_pred(dim_model, X_test, clf)

    dim_framework = st.sidebar.radio(
        "Choose interpretation framework", ["SHAP", "ELI5"]
    )

    ################################################
    # Model output
    ################################################
    if st.sidebar.checkbox("Preview uploaded data"):
        st.sidebar.dataframe(df.head())

    # the report is formatted to 2 decimal points (i.e. accuracy 1 means 1.00) dependent on streamlit styling update https://github.com/streamlit/streamlit/issues/1125
    st.sidebar.markdown("#### Classification report")
    show_perf_metrics(y_test, pred)

    ################################################
    # Global Interpretation
    ################################################
    st.markdown("#### Global Interpretation")
    st.text("Most important features")
    info_global = st.button("How it is calculated")
    if info_global:
        st.info(
            """
        The importance of each feature is derived from [permutation importance](https://www.kaggle.com/dansbecker/permutation-importance) -
        by randomly shuffle a feature, how much does the model performance decrease.
        """
        )
    # This only works if removing newline from html
    # Refactor this once added more models
    if dim_framework == "SHAP":
        show_global_interpretation_shap(X_train, clf)
    elif dim_framework == "ELI5":
        show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model)

    if st.sidebar.button("About the app"):
        st.sidebar.markdown(
            """
             Read more about how it works on [Github] (https://github.com/yanhann10/ml_interpret)
             Basic data cleaning recommended before upload   
             [Feedback](https://docs.google.com/forms/d/e/1FAIpQLSdTXKpMPC0-TmWf2ngU9A0sokH5Z0m-QazSPBIZyZ2AbXIBug/viewform?usp=sf_link)   
             Last update Mar 2020 by [@hannahyan](https://twitter.com/hannahyan)
              """
        )
        st.sidebar.markdown(
            '<a href="https://ctt.ac/zu8S4"><img src="https://image.flaticon.com/icons/svg/733/733579.svg" width=16></a>',
            unsafe_allow_html=True,
        )

    ################################################
    # Local Interpretation
    ################################################
    st.markdown("#### Local Interpretation")

    # misclassified
    if st.checkbox("Filter for misclassified"):
        X_test, y_test, pred = filter_misclassified(X_test, y_test, pred)
        if X_test.shape[0] == 0:
            st.text("No misclassification🎉")
        else:
            st.text(str(X_test.shape[0]) + " misclassified total")
            show_local_interpretation(
                X_test,
                y_test,
                clf,
                pred,
                target_labels,
                features,
                dim_model,
                dim_framework,
            )
    else:
        show_local_interpretation(
            X_test, y_test, clf, pred, target_labels, features, dim_model, dim_framework
        )

    ################################################
    # PDP plot
    ################################################
    if dim_model != "XGBoost" and st.checkbox("Show how features vary with outcome"):
        draw_pdp(clf, X_train, features, target_labels, dim_model)


if __name__ == "__main__":
    main()
