import pandas as pd

def normalise_database():
	df = pd.read_csv("feature_database.csv")
	columns = df.select_dtypes(include='number').columns
	mean = df.mean(numeric_only=True)
	std = df.std(numeric_only=True)
	df[columns] = (df[columns] - mean)/std
	df.to_csv("normalized_feature_database.csv")


normalise_database()