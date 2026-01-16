# src/main.py
from extract import extract_data
from transform import transform_data
from load import load_data

def run_etl():
    data = extract_data()
    transformed_data = transform_data(data)
    load_data(transformed_data)

if __name__ == "__main__":
    run_etl()



    