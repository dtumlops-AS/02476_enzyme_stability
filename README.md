# mlops_enzyme_stability

## Overal goal
The goal of this project is to attempt to predict the thermal stability of various enzymes given their amino acid sequence. The task was part of a competition hosted by Novozymes on Kaggle: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/overview

We intend to use the transformers framework from Huggingface.

The dataset is taken from https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data and consists of roughly 31 thousand samples. The samples consist of a unique sequence id, an amino acid sequence, a pH value that the amino acid was treated under, a data source (not used) and finally the prediction target, which is an unspecified continuous value where high values indicate a high thermal stability and vice versa.

The pretrained model we apply is a BERT model trained on amino acid sequences. The model was trained to finish a sequence of amino acids given the initial sequence. This task requires predicting a continuous value, so the BERT model is extended with a multi-layer perceptron.

Currently inference with the BERT model is computationally expensive, so unless we find a way to optimize this we will move on with another model
## Framework
We intend to use the transformers framework from Huggingface.

## Data
The dataset is taken from https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data and consist of roughly 31 thousand samples. The samples consist of a unique sequence id, an amino acid sequence, a pH value that the amino acid was treated under, a data source (not used) and finally the prediction target, which is an unspecified continuous value where high values indicate a high thermal stability and vice versa.
```
#To retrieve the data from google_storage (mediated through dvc) please just:
git clone https://github.com/AlexanderVoldby/02476_enzyme_stability.git -b dev_pau
cd 02476_enzyme_stability
dvc pull

```

## Models
The pretrained model we apply is a BERT model trained on amino acid sequences. The model was trained to finish a sequence of amino acids given the initial sequence. This task requires predicting a continuous value, so the BERT model is extended with a regression layer.

-------------------------------

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mlops_enzyme_stability  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
