import pandas as pd

def clean_dataset():
    """
    Cleans the initial dataset data/census.csv
    """

    source_data_location = "data/census.csv"
    cleaned_data_location = "data/census_clean.csv"

    # Read dataset skipping all leading and trailing whitespaces
    df = pd.read_csv(source_data_location, skipinitialspace=True)
    print("Data loaded from data/census.csv")

    # Remove all leading and trailing whitespaces from string (object) fields
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    print("Removed leading and trailing whitespaces from data values")

    initial_shape = df.shape
    print(f"Initial dataset shape: {initial_shape}")

    # Remove lines with unknown values marked by a question mark, as we won't do imputation
    for column in df.columns:
        df = df.drop(df[df[column] == "?"].index)

    processed_shape = df.shape
    print(f"Processed dataset shape: {processed_shape}")
    lines_removed = initial_shape[0] - processed_shape[0]
    print(f"{lines_removed} lines containing unknown values (?) removed ({lines_removed / initial_shape[0] * 100} %)")

    # Saving cleaned dataset to new file
    print("Cleaned data saved to data/census_clean.csv")
    df.to_csv(cleaned_data_location, index=False)


if __name__ == "__main__":
    clean_dataset()