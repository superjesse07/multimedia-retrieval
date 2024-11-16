import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine
from sklearn import metrics
import matplotlib.pyplot as plt
from functools import lru_cache
import ast

normalized_database_path = 'normalized_feature_database_final.csv'
full_database = pd.read_csv(normalized_database_path)

numeric_weights = np.array([0.2348, 0.0000, 0.0991, 0.1351, 0.2348, 0.1519, 0.1442])
histogram_weights = {'A3': 0.1774, 'D1': 0.0574, 'D2': 0.2351, 'D3': 0.3219, 'D4': 0.2082}
histogram_columns = list(histogram_weights.keys())
non_histogram_columns = full_database.select_dtypes(include=[np.number]).columns.difference(histogram_columns)

numeric_features = full_database[non_histogram_columns].values
numeric_spread = np.std(numeric_features, axis=0)

@lru_cache(maxsize=None)
def parse_histogram(hist_string):
    """Parse histogram data from a string to a numpy array."""
    if isinstance(hist_string, str):
        return np.array(ast.literal_eval(f"[{','.join(hist_string.split())}]")).flatten()
    return np.array(hist_string).flatten()

cached_histograms = {col: full_database[col].apply(parse_histogram).to_list() for col in histogram_columns}

true_table, score_table = {}, {}
precision = {}

for i in range(len(full_database)):
    query_category = full_database.loc[i, 'category']
    print(f'{i+1}/{len(full_database)}')    
    query_numeric = numeric_features[i].reshape(1, -1)
    db_numeric_indices = np.delete(np.arange(len(full_database)), i)  
    db_numeric = numeric_features[db_numeric_indices]

    normalized_query = (query_numeric / numeric_spread) * numeric_weights
    normalized_db = (db_numeric / numeric_spread) * numeric_weights
    cosine_distances = [cosine(normalized_query[0], row) for row in normalized_db]

    emd_distances = [
        sum(
            wasserstein_distance(cached_histograms[col][i], cached_histograms[col][j]) * histogram_weights[col]
            for col in histogram_columns
        )
        for j in db_numeric_indices
    ]

    combined_distances = np.array(cosine_distances) + np.array(emd_distances)

    if query_category not in true_table:
        true_table[query_category] = []
        score_table[query_category] = []
        precision[query_category] = []
    
    closest_indices = np.argsort(combined_distances)
    closest_categories = full_database.iloc[np.delete(np.arange(len(full_database)), i)[closest_indices]]['category']
    accuracy_10 = (closest_categories[:10] == query_category).sum() / 10
    precision[query_category].append(accuracy_10)
    true_table[query_category] += (full_database.iloc[db_numeric_indices]['category'] == query_category).tolist()
    score_table[query_category] += (-combined_distances).tolist()


categories, average_auc, top_10_accuracy = [], [], []
full_truth, full_score = [], []

for category in true_table:
    categories.append(category)
    fpr, tpr, _ = metrics.roc_curve(true_table[category], score_table[category])
    auc = metrics.roc_auc_score(true_table[category], score_table[category])
    average_auc.append(auc)

    top_10_accuracy.append(np.array(precision[category]).mean())

    full_truth += true_table[category]
    full_score += score_table[category]

    plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    plt.title(f"ROC for {category}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.savefig(f'auc/{category}.png')
    plt.clf()

fpr, tpr, _ = metrics.roc_curve(full_truth, full_score)
overall_auc = metrics.roc_auc_score(full_truth, full_score)

plt.plot(fpr, tpr, label=f"AUC={overall_auc:.2f}")
plt.title("ROC for All Categories")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.savefig('auc/full.png')
plt.clf()

categories.append('Overall')
average_auc.append(overall_auc)
overall_accuracy = sum(top_10_accuracy) / len(top_10_accuracy)
top_10_accuracy.append(overall_accuracy)

auc_summary = pd.DataFrame({'category': categories, 'average_auc': average_auc})
auc_summary.to_csv('auc_results.csv', index=False)

accuracy_summary = pd.DataFrame({'category': categories, 'top_10_accuracy': top_10_accuracy})
accuracy_summary.to_csv('accuracy_results.csv', index=False)

print("AUC results saved to 'auc_results.csv'")
print("Accuracy results saved to 'accuracy_results.csv'")
print(f"Overall AUC: {overall_auc:.4f}")
print(f"Overall Top-10 Accuracy: {overall_accuracy:.4f}")
