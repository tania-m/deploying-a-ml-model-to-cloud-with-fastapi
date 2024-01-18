import pandas as pd

def clean_dataset():
    """
    Cleans the initial dataset data/census.csv
    """

    # Read dataset skipping all leading and trailing whitespaces
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    print("Data loaded from data/census.csv")

    # Remove all leading and trailing whitespaces from string (object) fields
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

    initial_shape = df.shape
    print(f"Initial dataset shape: {initial_shape}")

    # Remove lines with unknown values marked by a question mark, as we won't do imputation
    for column in df.columns:
        df = df.drop(df[df[column] == "?"].index)

    processed_shape = df.shape
    print(f"Processed dataset shape: {processed_shape}")
    print(f"{initial_shape[0] - processed_shape[0]} lines containing unknown values (?) removed")

    # Saving cleaned dataset to new file
    print("Cleaned data saved to data/census_clean.csv")
    df.to_csv("data/census_clean.csv", index=False)


if __name__ == "__main__":
    clean_dataset()