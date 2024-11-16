import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import ast
from feature_extraction import process_single_query
from scipy.spatial.distance import cosine


normalized_database_path = "normalized_feature_database_final.csv"
full_database = pd.read_csv(normalized_database_path)


histogram_columns = ['A3', 'D2', 'D3', 'D4']
full_histogram_columns = ['A3', 'D1', 'D2', 'D3', 'D4']
feature_columns = full_database.select_dtypes(include=[np.number]).columns
non_histogram_columns = feature_columns.difference(full_histogram_columns)

stats_df = pd.read_csv("normalization_stats_final.csv", index_col=0)
mean = stats_df['mean']
std = stats_df['std']

def parse_histogram(hist_string):
    if isinstance(hist_string, str):
        cleaned_string = ','.join(hist_string.split())
        array = np.array(ast.literal_eval(f"[{cleaned_string}]"))
    elif isinstance(hist_string, (list, np.ndarray)):
        array = np.array(hist_string)
    else:
        raise ValueError("Unexpected format for histogram data")
    return array.flatten()

parsed_histograms = {col: full_database[col].apply(parse_histogram) for col in histogram_columns}

numeric_spread = full_database[non_histogram_columns].std().values
single_value_weights = 1 / numeric_spread  
single_value_weights /= single_value_weights.sum()  

histogram_spread = {
    hist: np.std([wasserstein_distance(h1, h2) 
                  for h1, h2 in zip(parsed_histograms[hist][:-1], parsed_histograms[hist][1:])])
    for hist in histogram_columns
}
histogram_weights = {hist: 1 / spread for hist, spread in histogram_spread.items()}
hist_sum = sum(histogram_weights.values())
histogram_weights = {k: v / hist_sum for k, v in histogram_weights.items()}  

def normalize_query(query_features):
    if query_features is None:
        raise ValueError("Query features could not be loaded. Please check the file path and file format.")
  
    numeric_query_features = {key: value for key, value in query_features.items() if isinstance(value, (int, float, np.float64))}
    normalized_query = pd.Series(numeric_query_features)
    normalized_query = (normalized_query - mean[normalized_query.index]) / std[normalized_query.index]

    histogram_features = {key: parse_histogram(query_features[key]) for key in histogram_columns if key in query_features}
    return {
        'numeric': normalized_query,
        'histograms': histogram_features
    }

def find_closest_entries(normalized_query, numeric_spread):

    normalized_database = full_database[non_histogram_columns]
    common_columns = normalized_database.columns.intersection(normalized_query['numeric'].index)
    normalized_query_num = (normalized_query['numeric'][common_columns] / numeric_spread) * single_value_weights

    normalized_db_values = (normalized_database[common_columns] / numeric_spread) * single_value_weights
    numeric_distances = [cosine(normalized_query_num.values, row) for row in normalized_db_values.values]

    emd_distances = []
    for row_idx, row in full_database.iterrows():
        emd_distance = sum(
            (wasserstein_distance(normalized_query['histograms'][hist], parsed_histograms[hist][row_idx]) / histogram_spread[hist]) * histogram_weights[hist]
            for hist in histogram_columns
        )
        emd_distances.append(emd_distance)

    combined_distances = numeric_distances + np.array(emd_distances)

    closest_indices = np.argsort(combined_distances)[:10]
    closest_entries = full_database.iloc[closest_indices][['category', 'file']]
    closest_entries = closest_entries.copy()
    closest_entries['cosine_distance'] = np.array(numeric_distances)[closest_indices]
    closest_entries['histogram_distance'] = np.array(emd_distances)[closest_indices]
    closest_entries['distance'] = combined_distances[closest_indices]

    print("Top 10 closest entries based on combined cosine and EMD distance:")
    print(closest_entries)
    
    return closest_entries

def query_obj(file_path):
    query_features = process_single_query(file_path)
    normalized_query = normalize_query(query_features)
    closest_entries = find_closest_entries(normalized_query, numeric_spread)
    return closest_entries

if __name__ == "__main__":
    query_obj("temp.obj")