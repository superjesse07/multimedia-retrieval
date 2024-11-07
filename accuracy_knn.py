import pandas as pd
import numpy as np
import faiss
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score

file_path = 'normalized_feature_database_final.csv'
full_database = pd.read_csv(file_path)

numeric_columns = full_database.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = full_database[numeric_columns].values.astype('float32')

histogram_columns = ['A3', 'D1', 'D2', 'D3', 'D4']

def parse_histogram_column(column_data):
    return np.array([
        np.fromstring(row.strip("[]").replace(" ", ","), sep=",") if isinstance(row, str) else np.array([])
        for row in column_data
    ], dtype=object)

parsed_histograms = [parse_histogram_column(full_database[col]) for col in histogram_columns]

max_length = max(len(hist) for column in parsed_histograms for hist in column if len(hist) > 0)
padded_histograms = [
    np.array([np.pad(hist, (0, max_length - len(hist)), constant_values=0) for hist in column], dtype='float32')
    for column in parsed_histograms
]

histogram_features_combined = np.hstack(padded_histograms)
combined_features = np.hstack((numeric_features, histogram_features_combined))

dimension = combined_features.shape[1]

def build_faiss_ann_index(features, dimension, nlist=100):
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.train(features)
    index.add(features)
    return index

faiss_index = build_faiss_ann_index(combined_features, dimension)

knn_model = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn_model.fit(combined_features)

def calculate_metrics(query_index, k=10):
    query_category = full_database.iloc[query_index]['category']
    query_vector = combined_features[query_index].reshape(1, -1)
    print(f"{query_index}")

    distances_ann, ann_indices = faiss_index.search(query_vector, k + 1)
    distances_knn, knn_indices = knn_model.kneighbors(query_vector, n_neighbors=k + 1)

    distances_ann = np.delete(distances_ann,0)
    ann_indices = np.delete(ann_indices,0)
    distances_knn = np.delete(distances_knn,0)
    knn_indices = np.delete(knn_indices,0)

    def calculate_scores(indices, distances):
        relevance_labels = full_database.iloc[indices]['category'] == query_category
        auc_score = roc_auc_score(relevance_labels, 1 / (distances.flatten() + 1e-10)) \
                    if len(set(relevance_labels)) > 1 else np.nan
        avg_precision = average_precision_score(relevance_labels, 1 / (distances.flatten() + 1e-10))
        precision_at_k = sum(relevance_labels) / k
        return auc_score, avg_precision, precision_at_k

    ann_auc, ann_avg_precision, ann_precision_at_k = calculate_scores(ann_indices, distances_ann)
    knn_auc, knn_avg_precision, knn_precision_at_k = calculate_scores(knn_indices, distances_knn)

    return {
        'ann_auc': ann_auc,
        'ann_avg_precision': ann_avg_precision,
        'ann_precision_at_k': ann_precision_at_k,
        'knn_auc': knn_auc,
        'knn_avg_precision': knn_avg_precision,
        'knn_precision_at_k': knn_precision_at_k,
        'category': query_category
    }

results = []
for i in range(len(full_database)):
    metrics = calculate_metrics(i, k=10)
    results.append(metrics)

results_df = pd.DataFrame(results)
numeric_metrics = results_df.select_dtypes(include=[np.number])

overall_metrics = numeric_metrics.mean()
category_metrics = results_df.groupby('category').mean()
category_metrics.to_csv('category_metrics.csv', index=True)

print("Overall metrics:", overall_metrics)
print("Category-wise metrics saved to 'category_metrics.csv'")
