import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/UCI_Credit_Card.csv")
PROCESSED_PATH = Path("data/processed/credit_data_processed.csv")

def prep_data():
    df = pd.read_csv(RAW_PATH)
    print(f'Размерность до очистки {df.shape}')
    
    df.columns = [col.strip().lower().replace('.', '_') for col in df.columns]
    df = df.dropna()
    
    print("Названия столбцов:", df.columns.tolist())
    Y = df["default_payment_next_month"]
    X = df.drop(columns=["default_payment_next_month"])
    
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    
    print(f"Предобработанные данные сохранены в {PROCESSED_PATH}")
    return X, Y

if __name__ == "__main__":
    prep_data()