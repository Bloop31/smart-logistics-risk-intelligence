import pandas as pd
import random

# Load CSV
df = pd.read_csv("smart_logistics_dataset.csv")

# Create 3 humidity bins (equal distribution)
df["humidity_bin"] = pd.qcut(df["Humidity"], 3, labels=[0, 1, 2])

# Assign rainfall based on bin
def assign_rainfall(bin_value):
    if bin_value == 0:
        return random.uniform(0, 5)
    elif bin_value == 1:
        return random.uniform(5, 15)
    else:
        return random.uniform(15, 40)

df["Precipitation(mm)"] = df["humidity_bin"].apply(assign_rainfall)

# Remove helper column
df.drop(columns=["humidity_bin"], inplace=True)

# Insert rainfall after humidity
cols = list(df.columns)
humidity_index = cols.index("Humidity")
cols.insert(humidity_index + 1, cols.pop(cols.index("Precipitation(mm)")))
df = df[cols]

# Overwrite same CSV file
df.to_csv("smart_logistics_dataset.csv", index=False)

print("Rainfall column added successfully!")
