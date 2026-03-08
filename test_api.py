import requests
import pandas as pd

df = pd.read_csv("/home/krishna/Personal/Major Project/car-price-backend/test_dataset/backend_test_dataset.csv")

url = "http://127.0.0.1:5000/predict"

for i,row in df.head(10).iterrows():

    payload = {
        "km_driven": row["km_driven"],
        "fuel": row["fuel"],
        "seller_type": row["seller_type"],
        "transmission": row["transmission"],
        "mileage": row["mileage"],
        "engine": row["engine"],
        "max_power": row["max_power"],
        "seats": row["seats"],
        "brand": row["brand"],
        "model": row["model"],
        "car_age": row["car_age"]
    }

    r = requests.post(url,json=payload)

    print(r.json())