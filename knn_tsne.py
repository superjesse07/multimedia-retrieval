import pandas as pd
import numpy as np
import faiss

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
    return index

def main():
    normalized_database_path = "normalized_feature_database.csv"
    faiss_ann_index = build_faiss_ann_index(normalized_database_path)

if __name__ == "__main__":
    main()
