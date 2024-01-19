# Model Card

[Template](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

- The model is based on a Random Forest Classifier.
- The random forest classifier here is used for binary classification.
- The classifier has been configured with n_estimators=50 (number of trees) and a random_state of 24.
- The model uses an encoder (OneHotEncoder) and a label binarizer (LabelBinarizer).
- The model has been created with sklearn 1.3.2.

## Intended Use

The model is intended to classify individuals in two income categories:

- equal or lower than 50k per year;
- larger than 50k per year.

## Training Data

- The data comes from a [dataset of the census bureau](https://archive.ics.uci.edu/dataset/20/census+income).
- We did not study the data distribution.
- The dataset contains anonymized demographic information.
- The original dataset has missing values (marked with question marks).
- Data lines with missing values were removed from the training dataset during cleaning/preprocessing.

## Evaluation Data

- The evaluation dataset was created by splitting the imported (pre-processed) data into train (80% of the dataset) and test datasets (20% dataset).

## Metrics

The model has been scored on 3 evaluation metrics:

- precision (true positives / (true positives + false positives)), with a valued performance of 0.73;
- recall (true positives / (true positives + false negatives)), with a valued performance of 0.60;
- f1 score (f beta score with beta = 1), with a valued performance of 0.67.

## Ethical Considerations

- The dataset is on the smaller side and should be used with care for predictions.
- Due to the small size of the dataset, it should be considered biased and not representative of a larger population.

## Caveats and Recommendations

- The data is outdated (Extraction was done from the 1994 Census database) and should be updated to reflect changes since that extraction date.
- The model has not been optimized (no hyperparameter tuning).
