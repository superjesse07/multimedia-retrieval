import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine, cdist
import ast
from functools import lru_cache
from sklearn import metrics

normalized_database_path = 'normalized_feature_database_final.csv'
full_database = pd.read_csv(normalized_database_path)

histogram_columns = ['A3', 'D2', 'D3', 'D4']  #No D1
feature_columns = full_database.select_dtypes(include=[np.number]).columns
non_histogram_columns = feature_columns.difference(['A3', 'D1', 'D2', 'D3', 'D4']) 

@lru_cache(maxsize=None)
def parse_histogram(hist_string):
    """Parse histogram data from a string to a numpy array."""
    if isinstance(hist_string, str):
        cleaned_string = ','.join(hist_string.split())
        array = np.array(ast.literal_eval(f"[{cleaned_string}]"))
    elif isinstance(hist_string, (list, np.ndarray)):
        array = np.array(hist_string)
    else:
        raise ValueError("Unexpected format for histogram data")
    return array.flatten()


def get_f_score(n,closest_categories,category):
    tp = (closest_categories[:n] == category).sum()
    fp = n-tp
    fn = (closest_categories[n:] == category).sum()
    return (2*tp)/(2*tp+fp+fn)

cached_histograms = {
    hist: full_database[hist].apply(parse_histogram).to_list()
    for hist in histogram_columns
}

single_value_weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
histogram_weights = {'A3': 0.2, 'D2': 0.2, 'D3': 0.2, 'D4': 0.2}  

numeric_features = full_database[non_histogram_columns].values
numeric_spread = np.std(numeric_features, axis=0)
histogram_spread = {
    hist: np.std([wasserstein_distance(cached_histograms[hist][i], cached_histograms[hist][j])
                  for i in range(len(cached_histograms[hist]))
                  for j in range(i + 1, len(cached_histograms[hist]))])
    for hist in histogram_columns
}

results = []

for i in range(len(full_database)):
    query_category = full_database.loc[i, 'category']
    print(query_category)

    query_numeric = numeric_features[i].reshape(1, -1)
    db_numeric = np.delete(numeric_features, i, axis=0)  

    spread_normalized_numeric_distances = cdist(
        (query_numeric / numeric_spread) * single_value_weights, 
        (db_numeric / numeric_spread) * single_value_weights, 
        metric='cosine'
    ).flatten()
    

    emd_distances = [
        sum(
            (wasserstein_distance(cached_histograms[hist][i], cached_histograms[hist][j]) / histogram_spread[hist]) * histogram_weights[hist]
            for hist in histogram_columns
        ) if i != j else np.inf 
        for j in range(len(full_database))
    ]
    emd_distances = np.delete(np.array(emd_distances), i) 

    combined_distances = spread_normalized_numeric_distances + emd_distances
    
    auc = metrics.roc_auc_score(full_database.iloc[np.delete(np.arange(len(full_database)), i)]['category'] == query_category,1/combined_distances)
    closest_indices = np.argsort(combined_distances)
    closest_categories = full_database.iloc[np.delete(np.arange(len(full_database)), i)[closest_indices]]['category']

    accuracy_20 = (closest_categories[:20] == query_category).sum() / 20
    accuracy_10 = (closest_categories[:10] == query_category).sum() / 10
    accuracy_5 = (closest_categories[:5] == query_category).sum() / 5
    results.append({
        'category': query_category,
        'accuracy_20': accuracy_20,
        'accuracy_10': accuracy_10,
        'accuracy_5': accuracy_5,
        'f_score_20': get_f_score(20,closest_categories,query_category),
        'f_score_10': get_f_score(10,closest_categories,query_category),
        'f_score_5': get_f_score(5,closest_categories,query_category),
        'auc': auc
    })

results_df = pd.DataFrame(results)

average_accuracy_per_class_20 = results_df.groupby('category')['accuracy_20'].mean()
average_accuracy_per_class_10 = results_df.groupby('category')['accuracy_10'].mean()
overall_accuracy_20 = results_df['accuracy_20'].mean()
overall_accuracy_10 = results_df['accuracy_10'].mean()
overall_accuracy_5 = results_df['accuracy_5'].mean()

average_f_score_per_class_20 = results_df.groupby('category')['f_score_20'].mean()
average_f_score_per_class_10 = results_df.groupby('category')['f_score_10'].mean()
overall_f_score_20 = results_df['f_score_20'].mean()
overall_f_score_10 = results_df['f_score_10'].mean()
overall_f_score_5 = results_df['f_score_5'].mean()

average_auc_per_class = results_df.groupby('category')['auc'].mean()
overall_auc = results_df['auc'].mean()

accuracy_summary = {
    'category': average_accuracy_per_class_20.index.tolist() + ['Overall'],
    'average_accuracy_top_20': average_accuracy_per_class_20.tolist() + [overall_accuracy_20],
    'average_accuracy_top_10': average_accuracy_per_class_10.tolist() + [overall_accuracy_10],
    'overall_accuracy_top_5': [None] * len(average_accuracy_per_class_20) + [overall_accuracy_5]
}
accuracy_summary_df = pd.DataFrame(accuracy_summary)

accuracy_summary_df.to_csv('accuracy_results_no_D1.csv', index=False)


f_score_summary = {
    'category': average_f_score_per_class_20.index.tolist() + ['Overall'],
    'average_f_score_top_20': average_f_score_per_class_20.tolist() + [overall_f_score_20],
    'average_f_score_top_10': average_f_score_per_class_10.tolist() + [overall_f_score_10],
    'overall_f_score_top_5': [None] * len(average_f_score_per_class_20) + [overall_f_score_5]
}
f_score_summary_df = pd.DataFrame(f_score_summary)

f_score_summary_df.to_csv('f_score_results_no_D1.csv', index=False)


auc_summary = {
    'category': average_auc_per_class.index.tolist() + ['Overall'],
    'average_auc': average_auc_per_class.tolist() + [overall_auc],}
auc_summary_df = pd.DataFrame(auc_summary)

auc_summary_df.to_csv('auc_results_no_D1.csv', index=False)

print("Overall accuracy for top 20 comparisons:", overall_accuracy_20)
print("Overall accuracy for top 10 comparisons:", overall_accuracy_10)
print("Overall accuracy for top 5 comparisons:", overall_accuracy_5)

print("Overall f_score for top 20 comparisons:", overall_f_score_20)
print("Overall f_score for top 10 comparisons:", overall_f_score_10)
print("Overall f_score for top 5 comparisons:", overall_f_score_5)

print("Overall auc:", overall_auc)

print("Category-specific accuracies saved to 'accuracy_results.csv'")
