# ML interpreter

Blackblox ML classifiers visually explained

## About

ML interpreter demonstrates auto-interpretability of machine learning models in a codeless environment.

Currently it focuses on high-performance blackbox tree ensemble models (random forest, XGBoost and lightGBM) for binary/multi-class classifications on tabular data, though the framework has the capability to extend to other models, other prediction types (regression), and other data types such as text/image.

It provides interpretation at both global and local levels:

- At global level, it indicates feature importance
- At local level, one can view how features affect individual predictions

## How it works

### Key features

- demo data/upload a small csv (a demo csv included in the github folder)
- choose among algorithms
- data preview and classification report
- global/local interpretations
- inspect misclassified data

To view how individual classification decision is made, one can toggle which datapoint to view.

<img src="ml_interpret.gif" alt='screenshot'>

Note: If preprocessing is needed, it is recommended to preprocess the data prior to the upload, since the app does not provide automatic data cleaning.

## How to run this demo

- [Demo app](https://ml-interpret.herokuapp.com/)

- Run from repo

```
git clone git@github.com:yanhann10/ml_interpret.git
cd ml_interpret
make install
streamlit run app.py
```

- Pull from Docker

```
docker pull yanhann10/ml-explained
streamlit run app.py
```

## Other resources

**Tutorials**
[ML Explainability by Kaggle](https://www.kaggle.com/learn/machine-learning-explainability)  
[Interpretable ML book](https://christophm.github.io/interpretable-ml-book/)

**Packages**
[SHAP](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27)  
[ELI5](https://eli5.readthedocs.io/en/latest/index.html)  
[PDPplot](https://pdpbox.readthedocs.io/en/latest/index.html)

[Feedback](https://docs.google.com/forms/d/e/1FAIpQLSdTXKpMPC0-TmWf2ngU9A0sokH5Z0m-QazSPBIZyZ2AbXIBug/viewform?usp=sf_link) welcomed
