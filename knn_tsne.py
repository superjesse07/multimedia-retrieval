import pandas as pd
import numpy as np
import faiss
from process_single_query import process_single_query
import ast

def parse_histogram(hist_string):
    """Parse histogram data from string or list/array."""
    if isinstance(hist_string, str):
        cleaned_string = ','.join(hist_string.split())
        array = np.array(ast.literal_eval(f"[{cleaned_string}]"))
    elif isinstance(hist_string, (list, np.ndarray)):
        array = np.array(hist_string)
    else:
        raise ValueError("Unexpected format for histogram data")
    return array.flatten()

def normalize_query(query_features):
    """Normalize the query features based on precomputed statistics."""
    stats_df = pd.read_csv("normalization_stats.csv", index_col=0)
    mean = stats_df['mean']
    std = stats_df['std']

    numeric_query_features = {key: value for key, value in query_features.items() if isinstance(value, (int, float, np.float64))}
    normalized_query = pd.Series(numeric_query_features)
    normalized_query = (normalized_query - mean[normalized_query.index]) / std[normalized_query.index]

    histogram_features = {key: parse_histogram(query_features[key]) for key in ['A3', 'D1', 'D2', 'D3', 'D4'] if key in query_features}
    
    return {
        'numeric': normalized_query,
        'histograms': histogram_features
    }

def build_faiss_ann_index(normalized_database_path, nlist=100):
    """Build a Faiss ANN index using the IVF Flat index."""
    full_database = pd.read_csv(normalized_database_path)
    feature_columns = full_database.select_dtypes(include=[np.number]).columns
    normalized_features = full_database[feature_columns].values.astype('float32')

    dimension = normalized_features.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    
    if normalized_features.shape[0] < 39 * nlist:
        nlist = max(1, normalized_features.shape[0] // 39)
        print(f"Warning: Adjusting nlist to {nlist} due to insufficient training points.")

    index.train(normalized_features)
    index.add(normalized_features)

    print(f"Faiss ANN index built with {normalized_features.shape[0]} vectors of dimension {dimension}.")
    return index, full_database, feature_columns

def query_faiss_ann_index(index, normalized_query_vector, k=5):
    """Query the Faiss ANN index to find the K-nearest neighbors of a feature vector."""
    normalized_query_vector = np.array(normalized_query_vector).astype('float32').reshape(1, -1)
    distances, indices = index.search(normalized_query_vector, k)
    return indices[0], distances[0]

def find_k_nearest_neighbors(query_features, normalized_database_path, k=10):
    """Find K-nearest neighbors using Faiss ANN index and display results."""
    normalized_query = normalize_query(query_features)
    
    # Build Faiss ANN index
    index, full_database, feature_columns = build_faiss_ann_index(normalized_database_path)
    
    # Numeric part of the query vector
    numeric_query_vector = normalized_query['numeric'][feature_columns].values
    
    # Find K-nearest neighbors using Faiss ANN index
    nearest_indices, distances = query_faiss_ann_index(index, numeric_query_vector, k=k)
    
    # Retrieve closest entries
    closest_entries = full_database.iloc[nearest_indices].copy()
    closest_entries['distance'] = distances

    print("Top closest entries based on ANN distance:")
    print(closest_entries[['category', 'file', 'distance']])
    
    return closest_entries

if __name__ == "__main__":
    query_file_path = r"normalised_v2_dataset/Car/D00168.obj"
    query_features = process_single_query(query_file_path)
    
    # Find and print the closest entries
    closest_entries = find_k_nearest_neighbors(query_features, "normalized_feature_database.csv", k=10)
