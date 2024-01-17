import pandas as pd

def clean_dataset():
    """
    Cleans the initial dataset data/census.csv
    """

    # Read dataset skipping all leading and trailing whitespaces
    df = pd.read_csv("data/census.csv")
    print("Data loaded from data/census.csv")
    
    # Remove all leading and trailing whitespaces from string (object) fields
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    
    # Saving cleaned dataset to new file
    print("Cleaned data saved to data/census_clean.csv")
    df.to_csv("data/census_clean.csv", index=False)


if __name__ == "__main__":
    clean_dataset()