import pandas as pd
import numpy as np
import faiss
from feature_extraction import process_single_query
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mplcursors


def build_faiss_ann_index(normalized_database_path, nlist=100):
    """Build a Faiss ANN index using the IVF Flat index."""
    full_database = pd.read_csv(normalized_database_path)

    histogram_columns = ['A3', 'D1', 'D2', 'D3', 'D4']
    full_database = flatten_histogram_features(full_database, histogram_columns)

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
    return index, full_database, normalized_features



def flatten_histogram_features(full_database, histogram_columns):
    """Flatten histogram columns and concatenate them with the numerical features."""
    flattened_histograms = []

    for col in histogram_columns:
        full_database[col] = full_database[col].str.replace(" ", ",")

        full_database[col] = full_database[col].str.replace(r',+', ',', regex=True)  
        full_database[col] = full_database[col].str.strip(",")  

        try:
            histogram_df = full_database[col].apply(eval).apply(pd.Series)
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            continue 

        histogram_df.columns = [f"{col}_{i}" for i in histogram_df.columns]
        flattened_histograms.append(histogram_df)

    full_database_flat = pd.concat([full_database.drop(columns=histogram_columns), *flattened_histograms], axis=1)
    
    return full_database_flat




colors_100 = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
    "#ff7f00", "#d2691e", "#ff1493", "#00ff7f", "#0000ff", "#ff6347", "#00ced1", "#2e8b57", "#6a5acd", "#ff4500",
    "#7fff00", "#da70d6", "#ff00ff", "#dc143c", "#8b0000", "#00fa9a", "#b03060", "#ffb6c1", "#deb887", "#556b2f",
    "#ff8c00", "#9932cc", "#b22222", "#ff69b4", "#2f4f4f", "#ffa500", "#228b22", "#cd5c5c", "#4682b4", "#d8bfd8",
    "#6b8e23", "#9acd32", "#483d8b", "#ff1493", "#adff2f", "#7fffd4", "#ff69b4", "#8fbc8f", "#b0c4de", "#7cfc00",
    "#ff4500", "#ba55d3", "#db7093", "#afeeee", "#9400d3", "#ffd700", "#dda0dd", "#ff4500", "#4682b4", "#ffdead",
    "#8b4513", "#bdb76b", "#e9967a", "#0000cd", "#8a2be2", "#da70d6", "#ff6347", "#40e0d0", "#ff1493", "#ff00ff",
    "#7cfc00", "#ff7f50", "#f08080", "#48d1cc", "#ff4500", "#ff1493", "#ff69b4", "#4169e1", "#b22222", "#ff6347",
    "#8a2be2", "#f5deb3", "#800080", "#ff6347", "#00bfff", "#8b0000", "#ff00ff", "#4b0082", "#ff1493", "#b03060",
    "#ffd700", "#ff7f50", "#d2691e", "#8a2be2", "#ff4500"
]

def find_k_nearest_neighbors(query_features, database_path, k=10):
    """Find k nearest neighbors for a given query in a Faiss index."""
    full_database = pd.read_csv(database_path)
    histogram_columns = ['A3', 'D1', 'D2', 'D3', 'D4']
    full_database = flatten_histogram_features(full_database, histogram_columns)

    feature_columns = full_database.select_dtypes(include=[np.number]).columns
    database_features = full_database[feature_columns].values.astype('float32')

    dimension = database_features.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(database_features)

    distances, indices = index.search(query_features.reshape(1, -1), k)
    neighbors = full_database.iloc[indices[0]]
    
    return neighbors


def perform_tsne_and_plot(full_database, features, colors_100, target_dim=2, perplexity=30, n_iter=700):
    """Perform t-SNE dimensionality reduction and plot the result."""
    tsne = TSNE(n_components=target_dim, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_features = tsne.fit_transform(features)
    shape_names = full_database['file']
    shape_classes = full_database['category']
    
    unique_classes = shape_classes.unique()
    num_classes = len(unique_classes)

    indices = [i for (i,x) in enumerate(shape_classes) if x in unique_classes]

    
    assert num_classes <= 100, "The number of categories exceeds the number of unique colors available."
    color_map = {cls: colors_100[i] for i, cls in enumerate(unique_classes)}
    
    color_values = shape_classes.map(color_map)
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    scatter = ax.scatter(
        reduced_features[indices, 0],
        reduced_features[indices, 1],
        c=color_values[indices],
        alpha=0.9
    )

    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
    
    handles = [plt.Line2D([0], [0], marker='o', color=color_map[cls], markersize=8, linestyle='', label=cls) 
               for cls in unique_classes]
    plt.legend(handles=handles, title="Shape Category", bbox_to_anchor=(1, 1), loc='upper left')
    
    plt.title("2D Scatter Plot of Shape Feature Vectors (t-SNE)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    cursor = mplcursors.cursor(hover=True)
    
    @cursor.connect("add")
    def on_hover(event):
        index = event.index
        name = shape_names.iloc[index]
        shape_class = shape_classes.iloc[index]
        event.annotation.set_text(f"File: {name}\nCategory: {shape_class}")
    
    plt.show()


def main():
    normalized_database_path = "normalized_feature_database_final.csv"
    
    faiss_ann_index, full_database, normalized_features = build_faiss_ann_index(normalized_database_path)
    perform_tsne_and_plot(full_database, normalized_features, colors_100)

if __name__ == "__main__":
    query_file_path = r"normalised_v2_dataset/Car/D00168.obj"
    query_features = process_single_query(query_file_path)
    #main()
    # Find and print the closest entries
    closest_entries = find_k_nearest_neighbors(query_features, "normalized_feature_database.csv", k=10)
    #main()