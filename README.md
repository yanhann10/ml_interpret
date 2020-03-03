# ML interpretor

Machine learning interpretability as-a-service

## About

ML interpretor is a demo for interpreting blackbox machine learning models in codeless environment.

It provides interpretation at global and local levels and visualization using partial dependence plot.

- At global level, it indicates feature importance
- At local level, one can view how features affect individual predictions

## How it works

One can try out demo data or upload a csv and select a ml algorithm to run automatically.

To view how individual classification decision is made, one can toggle which datapoint to view.

<img src="ml_interpret.gif" alt='screenshot'>

If preprocessing is needed, it is recommended to preprocess the data prior to the upload since automatic data cleaning is not part of the function.

## How to run this demo

- [Demo app](https://ml-interpret.herokuapp.com/)

- Run from repo

```
git clone git@github.com:yanhann10/ml_interpret.git
cd ml_interpret
make install
```

- Pull from Docker

```
docker pull yanhann10/ml-explained
```

Then to run the app locally

```
streamlit run app.py
```

## Other resources

[PDPplot](https://pdpbox.readthedocs.io/en/latest/index.html)  
[ELI5](https://eli5.readthedocs.io/en/latest/index.html)  
[Interpretable ML](https://christophm.github.io/interpretable-ml-book/)

(work-in-progress)
