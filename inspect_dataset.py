import pandas as pd

df = pd.read_csv("data/combined_dataset.csv")

print("Columns in dataset:")
print(df.columns)

print("\nSample data:")
print(df.head())

print("\nData info:")
print(df.info())
