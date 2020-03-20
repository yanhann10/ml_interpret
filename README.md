# ML interpretor

Machine learning interpretability as-a-service

## About

ML interpretor demonstraites auto-interpretability of machine learning models in a codeless environment.

Currently it focuses on high-performance blackbox models (random forest, XGBoost and lightGBM) for binary or multi-class classifications on tabular data, though the framework has the capability to extend to other sklearn or keras model beyond boosted trees, and on other data types such as text data.

It provides interpretation at global and local levels and shows how features influence the outcome.

- At global level, it indicates feature importance
- At local level, one can view how features affect individual predictions

## How it works

### Key features

- demo data/upload a small csv (a demo csv included in the github folder)
- choose among algorithms
- data preview and classification report
- global/local interpretation
- inspect misclassified data

To view how individual classification decision is made, one can toggle which datapoint to view.

<img src="ml_interpret.gif" alt='screenshot'>

_Note: If preprocessing is needed, it is recommended to preprocess the data prior to the upload since automatic data cleaning is not part of the function. _

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

[PDPplot](https://pdpbox.readthedocs.io/en/latest/index.html)  
[ELI5](https://eli5.readthedocs.io/en/latest/index.html)  
[Interpretable ML book](https://christophm.github.io/interpretable-ml-book/)

[Feedback](https://docs.google.com/forms/d/e/1FAIpQLSdTXKpMPC0-TmWf2ngU9A0sokH5Z0m-QazSPBIZyZ2AbXIBug/viewform?usp=sf_link) welcomed
