from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from data import process_data


class MissingModelException(Exception):
    # Placeholder exception to have a named exception
    pass


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


def compute_slice_performance(model, encoder, lb, categorical_features, slice_features, processed_df, target_label):
    """ Compute performance on slices of data 
        for categorical features
        
        Inputs
        ------
        model : trained machine learning model.
        encoder : OneHot categorical encoder
        lb : label binarizer
        categorical_features: list of categorical 
            features
        slice_features: feature(s) to slice on
        processed_df: dataframe to use for slice
            performance computing (not pre-processed,
            process_data will be recalled here)
        targer_label: Name of the label column in `X`
            for process_data
        Returns
        -------
        Slicing performance evaluation results
    """

    # Results placeholder
    slice_details = []

    # In case we got only one feature
    if not isinstance(slice_features, list):
        slice_features = [slice_features]

    # Go over list of feature slices
    for feature in slice_features:
        # For the available categorical values...
        for value in processed_df[feature].unique():
            X_slice = processed_df[processed_df[feature] == value]
            # Prepare dataset
            X_slice, y_slice, _, _ = process_data(
                X_slice,
                categorical_features,
                label=target_label,
                training=False,
                encoder=encoder,
                lb=lb)
            # Predictions and model evaluation
            y_preds = inference(model, X_slice)
            precision, recall, f1_score = compute_model_metrics(y_slice, y_preds)
            # Keep results
            slice_details.append([feature, value, precision, recall, f1_score])

    # Write results to file
    slice_output_filename = "slice_output.txt"
    with open(slice_output_filename, "w") as slice_results_file:
        for row_item in slice_details:
            slice_results_file.write(f"{row_item[0]}, {row_item[1]}: {row_item[2]}, {row_item[3]}, {row_item[4]}\n")
    
    return slice_details


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
