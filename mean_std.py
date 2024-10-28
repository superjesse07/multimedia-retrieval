import pandas as pd

# Load the normalized CSV file
df_normalized = pd.read_csv("feature_database.csv")

# Assume the normalized data has a mean close to zero and a variance close to one
estimated_mean = df_normalized.mean()  # Should be approximately zero for normalized data
estimated_std = df_normalized.std()    # Should be approximately one for normalized data

# If the normalized mean is very close to zero, we interpret that the original mean is zero
# Otherwise, the original `std` and `mean` cannot be accurately retrieved without the original data.
print("Estimated Mean (should be close to 0):")
print(estimated_mean)
print("Estimated Std (should be close to 1):")
print(estimated_std)
