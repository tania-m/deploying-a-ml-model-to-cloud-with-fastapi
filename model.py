from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class MissingModelException(Exception):
    pass


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    
    # We want a model that:
    # - can deal with numerical (age) and categorical features (eg. workclass)
    # - does not make assumptions about the data distribution
    #   (as we did not spend a lot of time analyzing it)
    # - is not influenced too much by possible outliers
    #   (as we didn't look for or remove outliers in the datasets)
    # - could do regression or classification (for flexibility)
    # - should perfomr well even with missing values 
    #   (we cleaned the dataset to remove missing values, so this
    #   could come in handy if we decide later on to actually
    #   keep the missing values instead of removing them)

    # Define random forest parameters
    number_of_trees = 50 # default is 100
    random_seed = 24 # for reproduceable runs
    # max_depth = 150 # reduces risk of overfitting
    
    # Initialize RandomForestClassifier
    print("Using Random Forest Classifier")
    model = RandomForestClassifier(
        n_estimators=number_of_trees,
        random_state=random_seed,
        verbose=1)
    
    # Fitting model
    print("Fitting model")
    model.fit(X_train, y_train)
    print("Model fitting DONE!")
    
    return model


def compute_model_metrics(y, preds, beta_value=1):
    """
    Validates the trained machine learning model 
    using precision, recall, and fbeta.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    
    # true positives / (true positives + false positives)
    precision = precision_score(y, preds, zero_division=1)
    
    # true positives / (true positives + false negatives)
    recall = recall_score(y, preds, zero_division=1)
    
    # contrary to f1, fbeta defines a weight 
    # to balance between precision and recall using the beta parameter
    # when beta_value = 1, we have the F1 score
    fbeta = fbeta_score(y, preds, beta=beta_value, zero_division=1)
    # closer to 1 is better for fbeta
    
    return precision, recall, fbeta


def compute_slice_performance():
    """ Compute performance on slices of data 
        for categorical features
    """


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Random Forest Classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    
    if model is not None:
        return model.predict(X)
    
    # model is not set
    raise MissingModelException("Cannot make predictions without a model")
