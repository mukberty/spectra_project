import pickle
import pandas as pd

FILE = "data/1.pkl"

print("Opening file...")

with open(FILE, "rb") as f:
    data = pickle.load(f)

print("\nTYPE:\n")
print(type(data))

if isinstance(data, pd.DataFrame):

    print("\nCOLUMNS:\n")
    print(data.columns)

    print("\nHEAD:\n")
    print(data.head())

    print("\nSHAPE:\n")
    print(data.shape)

elif isinstance(data, dict):

    print("\nKEYS:\n")
    print(data.keys())

    first_key = list(data.keys())[0]

    print("\nFIRST ITEM:\n")
    print(data[first_key])

elif isinstance(data, list):

    print("\nLIST LENGTH:\n")
    print(len(data))

    print("\nFIRST ITEM:\n")
    print(data[0])

else:
    print(data)
