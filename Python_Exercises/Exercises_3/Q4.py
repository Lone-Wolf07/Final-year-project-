import pandas as pd

filename = input("Enter the CSV filename: ")
data = pd.read_csv(filename)

data = data.dropna()
data = data.drop_duplicates()

print("Summary statistics:")
print(data.describe())
