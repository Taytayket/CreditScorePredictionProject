import pandas as pd
from data_process import clean_and_engineer

def ppl(filepath: str) -> pd.DataFrame:
    try:
        print(f"Extracting data from {filepath}...")
        df_raw = pd.read_csv(filepath)

        print("Transforming data...")
        df_transformed = clean_and_engineer(df_raw)

        print("ETL completed. Sample:")
        print(df_transformed.head())

        return df_transformed
    
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    
    except Exception as e:
        print(f"ETL process failed: {e}")
        return pd.DataFrame()