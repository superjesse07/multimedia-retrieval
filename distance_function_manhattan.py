import pandas as pd
import numpy as np
from process_single_query import process_single_query
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cityblock  
import ast

query_file_path = r"C:\Users\Carlijn\multimedia-retrieval-3\normalised_v2_dataset\Bird\D00546.obj"
query_features = process_single_query(query_file_path)
print("query_features keys:", query_features)

def parse_histogram(hist_string):
    if isinstance(hist_string, str):
        cleaned_string = ','.join(hist_string.split())
        array = np.array(ast.literal_eval(f"[{cleaned_string}]"))
    elif isinstance(hist_string, (list, np.ndarray)):
        array = np.array(hist_string)
    else:
        raise ValueError("Unexpected format for histogram data")
    return array.flatten()

def normalize_query(query_features):
    stats_df = pd.read_csv("normalization_stats.csv", index_col=0)
    mean = stats_df['mean']
    std = stats_df['std']

    numeric_query_features = {key: value for key, value in query_features.items() if isinstance(value, (int, float, np.float64))}
    normalized_query = pd.Series(numeric_query_features)
    normalized_query = (normalized_query - mean[normalized_query.index]) / std[normalized_query.index]
    print("normalized_query:", normalized_query)

    histogram_features = {key: parse_histogram(query_features[key]) for key in ['A3', 'D1', 'D2', 'D3', 'D4'] if key in query_features}
    print("histogram_features:", histogram_features)
    
    return {
        'numeric': normalized_query,
        'histograms': histogram_features
    }

normalized_query = normalize_query(query_features)
print("Normalized query histograms:", normalized_query['histograms'].keys())

def find_closest_entries(normalized_query, normalized_database_path):

    full_database = pd.read_csv(normalized_database_path)
    
    histogram_columns = ['A3', 'D1', 'D2', 'D3', 'D4']
    feature_columns = full_database.select_dtypes(include=[np.number]).columns
    non_histogram_columns = feature_columns.difference(histogram_columns)
    normalized_database = full_database[non_histogram_columns]
 
    common_columns = normalized_database.columns.intersection(normalized_query['numeric'].index)
    normalized_query_num = normalized_query['numeric'][common_columns]
    normalized_database = normalized_database[common_columns]

    manhattan_distances = normalized_database.apply(
        lambda row: cityblock(row, normalized_query_num), axis=1
    )

    emd_distances = []
    for i, row in full_database.iterrows():
        emd_distance = 0
        for hist in histogram_columns:
            query_hist = normalized_query['histograms'][hist]
            row_hist = parse_histogram(row[hist])
            emd_distance += wasserstein_distance(query_hist, row_hist)
        emd_distances.append(emd_distance)

    combined_distances = pd.Series(emd_distances) * 600 + manhattan_distances 

    closest_indices = combined_distances.nsmallest(10).index
    closest_entries = full_database.loc[closest_indices, ['category', 'file']]
    closest_entries = closest_entries.copy()
    closest_entries['manhattan_distance'] = manhattan_distances.loc[closest_indices].values
    closest_entries['histogram_distance'] = pd.Series(emd_distances).loc[closest_indices].values
    closest_entries['combined_distance'] = combined_distances.loc[closest_indices].values

    print("Top 10 closest entries based on combined Manhattan and EMD distance:")
    print(closest_entries)
    
    return closest_entries

closest_entries = find_closest_entries(normalized_query, "normalized_feature_database.csv")
