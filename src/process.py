import numpy as np
import pandas as pd
import sys
df = pd.read_csv("../data/listings.csv")
out = r"C:\Users\mertc\Desktop\vienna_airbnb\data\extracted.csv"
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

def print_data(path):
    data = pd.read_csv(path)
    print(f"{data.head(10)}")
    print("*******************************")
    print(f"{data.info()}")
    print("*******************************")
    print(data.describe(include='all'))

def extract():

    # using 8 inputs, latitude & longitude for location,
    # room type (will be encoded)
    # minimum_nights
    # number of reviews
    # reviews per month
    # calculated host listings count
    # availability_365
    # price for output ofc.
    input = [
        "latitude",
        "longitude",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "price"
    ]
    # one-hot encoding map for room_type.
    # "Entire home/apt" room_Entire home/apt = 1
    # "Hotel room"     room_Hotel room = 1
    # "Private room"    room_Private room = 1
    # "Shared room"     room_Shared room = 1

    data = df[input].copy()
    data = pd.get_dummies(data, columns=["room_type"], prefix="room") # 0ne hot encoding 1 for true 0 for fasle
    missing = [c for c in input if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    data.to_csv(out, index=False)

def normalize(path):
    data = pd.read_csv(path)
    numeric_cols = [
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "price"
    ]

    for col in numeric_cols:
        min_val = data[col].min()
        max_val = data[col].max()
        data[col] = (data[col] - min_val) / (max_val - min_val)

    data.to_csv(out, index=False)

def split(path):
    from sklearn.model_selection import train_test_split
    data = pd.read_csv(path)
    x = data.drop(columns=["price"]).values
    y = data["price"].values

    #split 80% train, 20% test. traditional
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, test_size=0.2, shuffle=True, random_state=69
    )
    return x_train, x_test, y_train, y_test

#extract()
#print_data(out)
#normalize(out)
#split(out)